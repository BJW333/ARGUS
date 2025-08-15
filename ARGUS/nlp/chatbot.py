import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import socket
import requests
#from nlp.attention import BahdanauAttention
from nlp.reward import DynamicRewardSystem
#from nlp.intent import IntentClassifier
from config import script_dir, nlp
import subprocess
import time
import language_tool_python
from transformers import AutoTokenizer, pipeline
import torch
import platform
from metrics.logging import log_debug #, log_metrics

class Chatbot:
    def __init__(self, model_id="llama3-8b-q4_0-argus", max_tokens=400, max_generate_attempts: int = 10):
        self.model_name = model_id
        self.max_tokens = max_tokens
        self.ollama_host = "localhost"
        self.ollama_port = 11434
        
        self.max_generate_attempts = max_generate_attempts #max number of attempts to generate a response
        
        self.max_paraphrase_tokens = 128 #maximum length for paraphrasing
    
        #init the paraphraser pipeline
        #device = 0 if tf.config.list_physical_devices('GPU') else -1
        if platform.system() == "Linux":
            device = 0 if torch.cuda.is_available() else -1
        elif platform.system() == "Windows":
            device = 0 if torch.cuda.is_available() else -1
        elif platform.system() == "Darwin":  # macOS
            device = "mps" if torch.backends.mps.is_available() else -1 
            
        tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", legacy=False)
        self.paraphraser = pipeline(
            "text2text-generation",
            model="Vamsi/T5_Paraphrase_Paws",
            tokenizer=tokenizer,
            device=device
        )
        #print("device being used for paraphrase:", device)  # Debugging line to see which device is used
        self.enable_grammar_correction = True # Enable grammar correction by default

        self.ensure_ollama_running()
        log_debug("[ARGUS LLM] Model ready.")
        
        self.nlp = nlp
        self._doc_cache = {} # Cache for processed stuff
        self.embedding_cache = {} # Cache for embeddings
        
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')        
        self.reward_system = DynamicRewardSystem()
        #self.intent_classifier = IntentClassifier() #we dont use intent match anymore for confidence calculation due to inaccuracy
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
    
    def _get_doc(self, text: str):
        """
        Get a spaCy Doc object for the given text, caching results to avoid recomputation.
        """
        if text not in self._doc_cache:
            self._doc_cache[text] = self.nlp(text)
        return self._doc_cache[text]
    
    def get_embedding(self, text):
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.semantic_model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,       # turn off tqdm
            )
        return self.embedding_cache[text]
    
    def is_ollama_running(self):
        try:
            with socket.create_connection((self.ollama_host, self.ollama_port), timeout=2):
                return True
        except socket.error:
            return False
        
    def wait_for_ollama_ready(self, timeout=20):
        start = time.time()
        while time.time() - start < timeout:
            if self.is_ollama_running():
                return True
            time.sleep(1)
        raise RuntimeError("Ollama failed to start in time.")
    
    def ensure_ollama_running(self):
        if not self.is_ollama_running():
            log_debug(f"[ARGUS LLM] Ollama not running. Starting model '{self.model_name}'...")
            subprocess.Popen(["ollama", "run", self.model_name])
            self.wait_for_ollama_ready()  # Give Ollama time to initialize
        else:
            log_debug("[ARGUS LLM] Ollama is already running.")
    
    def calculate_confidencevalue_response(self, user_input, bot_response, candidates):
        """
        Calculate confidence as a weighted sum of:
            - S: semantic similarity to user input
            - D: diversity from other candidates
            - R: reward system score the (evaluate_response) function is what is used 
        All values are normalized to [0,1]
        The weights for S, D, and R can be adjusted based on the importance of each factor.
        Weights can be tuned to your needs.
        Parameters:
            user_input (str): The user's input text.
            bot_response (str): The bot's response text.
            candidates (list): List of candidate responses to compare against.
        Returns:
            final_confidence (float): The final confidence score, normalized to [0, 1].
            factors (dict): Dictionary containing the individual factors S, D, and R.
        """
        #Reward System score
        response_reward_score = self.reward_system.evaluate_response(user_input, bot_response) 
        #print(f"This is the raw R value: {response_reward_score}") #debugging line
        #normalize R to be between 0 and 1
        R = (response_reward_score + 45) / 90
        
        #Semantic Similarity (S)
        user_embedding = self.get_embedding(user_input)
        bot_embedding = self.get_embedding(bot_response)
        S = util.pytorch_cos_sim(user_embedding, bot_embedding).item()

        #Response Diversity Factor (D)
        candidate_embeddings = [self.get_embedding(c) for c in candidates]
        
        similarities = [
            util.pytorch_cos_sim(bot_embedding, c).item()
            for c in candidate_embeddings
            if not torch.allclose(bot_embedding, c, atol=1e-4)
        ]
        avg_similarity = np.mean(similarities) if similarities else 0.0
        D = 1 - avg_similarity  #Lower similarity = higher diversity
        
        
        #normalize S, D, R to be between 0 and 1
        S = max(0, min(1, S)) #ensure S is between 0 and 1
        D = max(0, min(1, D)) #ensure D is between 0 and 1
        R = max(0, min(1, R)) #ensure R is between 0 and 1
        
        #Weights for each factor
        # These weights can be adjusted based on the importance of each factor in your application.
        # For example, you could use a machine learning model to learn the optimal weights based on
        # historical data or user feedback.
        # Here, we use a simple linear combination of the factors with fixed weights.
        alpha = 0.5    # semantic similarity
        beta  = 0.2    # diversity
        gamma = 0.3    # reward/fitness

        final_confidence = np.clip(alpha * S + beta * D + gamma * R, 0, 1)
        log_debug(f"Final confidence score after np.clip: {final_confidence}") #debugging line
        
        #return the final confidence score and the individual factors
        factors = {"S": S, "D": D, "R": R} # Debugging line to see individual factors
        log_debug(f"Factors: {factors}")  # Debugging line to see individual factors
        
        return final_confidence        
        
    
    def is_semantically_similar(self, new_cand, existing_cands, threshold=0.85):
        new_emb = self.get_embedding(new_cand)
        for existing in existing_cands:
            existing_emb = self.get_embedding(existing)
            if util.pytorch_cos_sim(new_emb, existing_emb).item() > threshold:
                return True
        return False
    
    def generate_ARGUS_llmresponse(self, input_sentence, memory_prefix, num_candidates=5, temperature=0.75, top_k=30, top_p=0.9): #may have lower num_canidates to 3
        attempts = 0
        
        #prompt engineering
        log_debug("This is the input_sentence:\n", input_sentence) #debugging line to see input sentence
        log_debug("\nThis is the memory_prefix:\n", memory_prefix) #debugging line to see memory prefix
        prompt = memory_prefix + input_sentence
        log_debug("\nThis is what the prompt looks like:\n", prompt) #debugging line
        
        #collect multiple candidate responses from generation
        unique_candidates = set()
        all_candidates = []
        while len(all_candidates) < num_candidates and attempts < self.max_generate_attempts:
            attempts += 1
            try:
                response = requests.post(
                    f"http://{self.ollama_host}:{self.ollama_port}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": self.max_tokens,
                            "top_k": top_k,
                            "top_p": top_p
                        }
                    },
                    timeout=30
                )
                response.raise_for_status()
                candidate = response.json().get("response", "").strip()
                
                #postprocess token ids into a string
                if candidate not in unique_candidates and not self.is_semantically_similar(candidate, all_candidates):
                   unique_candidates.add(candidate)
                   all_candidates.append(candidate)
                   log_debug("New candidate:", candidate)
                else:
                   log_debug("Duplicate candidate detected. Regenerating...")
                
            except requests.exceptions.RequestException as e:
                log_debug(f"[ARGUS LLM] Error during generation: {e}")
                return ["⚠️ ARGUS encountered a communication error…"]
    
        #return the list of candidate strings
        #print("This is all canidates returned from the llm:", all_candidates) #debugging line
        
        return all_candidates

    
    def grammar_correct(self, text: str) -> str:
        """
        Perform grammar correction on the text using LanguageTool.
        """
        matches = self.grammar_tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    
    def crossover_text_advanced(self, parent_a: str, parent_b: str) -> str:
        """
        Perform parse-tree–based crossover by extracting top-level clauses
        from each parent, then splicing them.

        1) Parse each parent's text with spaCy.
        2) Extract clauses with 'extract_clauses'.
        3) Randomly choose half from each and combine.
        4) Optionally reorder them or do grammar check at the end.
        """
        
        def subtree_text(root_token):
            """
            Return the text of an entire subtree from 'root_token',
            including punctuation tokens that fall under it in the parse tree.
            """
            #collect all tokens in the subtree
            tokens = list(root_token.subtree)
            #sort them by their position in doc
            tokens = sorted(tokens, key=lambda t: t.i)
            #join their text
            return " ".join(t.text for t in tokens)
            
        def extract_clauses(doc):
            """
            Extract top-level clauses (root subtrees) from a spaCy doc.
            For each sentence, we grab the subtree of the main root.
            We also handle certain conj or coordinating roots for multiple clauses.
            """
            clauses = []
            seen = set()  # Track unique clauses
            for sent in doc.sents:
                root = sent.root
                clause_text = subtree_text(root).strip()
                if clause_text and clause_text not in seen:
                    clauses.append(clause_text)
                    seen.add(clause_text)
                for token in sent:
                    #optionally look for coordinated roots or conj
                    #eg "He ran and he jumped"
                    #we can gather other 'conj' heads that match the root or sentence boundary
                    if token.dep_ == "conj" and token.head == root:
                        conj_text = subtree_text(token).strip()
                        if conj_text and conj_text not in seen:
                            clauses.append(conj_text)
                            seen.add(conj_text)
            return clauses
        
        
        doc_a = self._get_doc(parent_a)
        doc_b = self._get_doc(parent_b)

        clauses_a = extract_clauses(doc_a)  # list of strings
        clauses_b = extract_clauses(doc_b)

        if not clauses_a and not clauses_b:
            return parent_a  #fallback if both empty
        if not clauses_a:
            return parent_b
        if not clauses_b:
            return parent_a

        #pick half from A and half from B
        half_a = random.sample(clauses_a, k=max(1, len(clauses_a)//2))
        half_b = random.sample(clauses_b, k=max(1, len(clauses_b)//2))

        child_clauses = half_a + half_b
        #random.shuffle(child_clauses)  #optional shuffle

        #Duplicate Removal Only When Needed
        if len(set(child_clauses)) < len(child_clauses):  #check if duplicates exist
            child_clauses = list(dict.fromkeys(child_clauses))  #remove duplicates efficiently

        child_text = " ".join(child_clauses)
        log_debug("Child text uncleaned:", child_text) #debugging line
        
        #return the final child text
        return child_text


    #----------------------------------------------------------------------------------------

    def mutate_semantic_drift(self, base_text: str, num_variants: int = 5, sim_range=(0.75, 0.95)) -> str:
        """
        Generate multiple paraphrased variants of the input text.
        Pick one that differs just enough to be useful for exploration.
        """
        paraphrases = self.paraphraser(
            f"paraphrase: {base_text}",
            max_length=self.max_paraphrase_tokens,
            num_return_sequences=num_variants,
            num_beams=max(4, num_variants),
            clean_up_tokenization_spaces=True,
        )

        original_emb = self.get_embedding(base_text)
        viable = []

        for item in paraphrases:
            cand = item["generated_text"].strip()
            cand_emb = self.get_embedding(cand)
            sim = util.pytorch_cos_sim(original_emb, cand_emb).item()
            
            if sim_range[0] < sim < sim_range[1]:  # not too close, not too far
                viable.append((cand, sim))

        if not viable:
            return base_text  # fallback

        # Pick the one with lowest similarity (maximum drift within bounds)
        best_variant = sorted(viable, key=lambda x: x[1])[0][0]
        
        log_debug(f"Best semantic drift variant: {best_variant}") #debugging line
        
        return best_variant
    
    
    def semantic_sim(self, a: str, b: str):
        e1 = self.get_embedding(a)
        e2 = self.get_embedding(b)
        return util.pytorch_cos_sim(e1, e2).item()
    
    def select_parents(self, population, scores, population_size):
        ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:population_size // 2]]

    def evolve_population(self, parents, population_size, crossover_rate, mutation_rate):
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choice(parents), random.choice(parents)
            child = parent1
            if random.random() < crossover_rate and self.semantic_sim(parent1, parent2) <= 0.95:
                child = self.crossover_text_advanced(parent1, parent2)
            if random.random() < mutation_rate:
                child = self.mutate_semantic_drift(child)
            new_population.append(child)
        return new_population    
                
    def ga_rerank_candidates(self, user_input, candidates,
                            pop_size=6, generations=2,
                            crossover_rate=0.5, mutation_rate=0.2): 
        #orginal values were pop_size=10, generations=4, crossover_rate=0.5, mutation_rate=0.3
        """
        1) Start with 'candidates' as the initial population.
        2) Evaluate them with 'reward_system' as the fitness.
        3) Evolve for 'generations' times.
        4) Return the best final string.
        """
        # ensure population has at least pop_size
        candidates = list(set(candidates))  # Remove duplicates
        population = candidates[:pop_size]
        
        log_debug("population:", population) #debugging line to see initial population
        log_debug("candidates:", candidates) #debugging line to see initial candidates
        
        while len(population) < pop_size:
            base = random.choice(candidates)
            population.append(self.mutate_semantic_drift(base))
        
        def fitness_func(candidate_text):
            # now fitness == confidence
            return self.calculate_confidencevalue_response(user_input, candidate_text, population)

        #Evaluate initial population
        scores = np.array([fitness_func(c) for c in population]) #new optimized line with nummpy less memory useage delete line and use one above if issue

        for gen in range(generations):
            # Selection: pick top half as parents
            parents = self.select_parents(population, scores, pop_size)
            population = self.evolve_population(
                parents,
                population_size=pop_size,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate
            )
            scores = np.array([fitness_func(c) for c in population])

        # Finally, pick the best
        confidences = list(zip(population, scores))
        best_candidate, best_confidence = max(confidences, key=lambda x: x[1])

        if self.enable_grammar_correction:
            best_candidate = self.grammar_correct(best_candidate)

        log_debug(f"[GA Final] Best candidate (confidence = {best_confidence:.2f}): {best_candidate}") #debugging line
        return best_candidate, best_confidence, confidences  # now confidences is list of (cand,conf)
                
