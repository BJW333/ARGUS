import tensorflow as tf
import numpy as np
import random
import re
from sentence_transformers import SentenceTransformer, util
import spacy
import string
import os

from nlp.attention import BahdanauAttention
from nlp.reward import DynamicRewardSystem
from nlp.intent import IntentClassifier
from config import script_dir, nlp


class Seq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Seq2SeqModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.encoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True, dropout=0.5)
        self.attention = BahdanauAttention(hidden_units)
        self.decoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True, dropout=0.5)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        #maybe multihead attention here instead of BahdanauAttention might be extremely beneficial from a attention standpoint
        #import torch.nn as nn
        #self.attention = nn.MultiheadAttention(hidden_units)
        
        
    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_embeddings = self.embedding(encoder_inputs)
        encoder_outputs, state_h, state_c = self.encoder(encoder_embeddings)
        context_vector, _ = self.attention(state_h, encoder_outputs)
        decoder_embeddings = self.embedding(decoder_inputs)

        #repeat the context vector across the sequence length
        repeated_context_vector = tf.repeat(tf.expand_dims(context_vector, 1), repeats=decoder_inputs.shape[1], axis=1)

        #decoder embeddings with the repeated context vector
        decoder_input_with_context = tf.concat([decoder_embeddings, repeated_context_vector], axis=-1)

        #pass the concatenated input to the decoder
        decoder_outputs, _, _ = self.decoder(decoder_input_with_context, initial_state=[state_h, state_c])
        logits = self.fc(decoder_outputs)
        return logits


class Chatbot:
    def __init__(self, vocab_size, embedding_dim, hidden_units, tokenizer, start_token, end_token, max_length):
        #self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #maybe remove these
        #self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2') #maybe remove these
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.vocab_size = vocab_size
        self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.optimizer = tf.keras.optimizers.Adam()

        #new inputs below
        self.nlp = nlp
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')        
        self.sentiment_analyzer = DynamicRewardSystem()
        self.intent_classifier = IntentClassifier()  
        
    def preprocess_sentence(self, sentence):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))  #remove punctuation
        encoded_sentence = [self.tokenizer.encode(self.start_token)[0]] + self.tokenizer.encode(sentence) + [self.tokenizer.encode(self.end_token)[0]]
        #ensure the sentence does not exceed max_length
        encoded_sentence = encoded_sentence[:self.max_length]
        #pad the sentence to max_length
        encoded_sentence = encoded_sentence + [0] * (self.max_length - len(encoded_sentence))
        return encoded_sentence

    #deletes start and end tokens and does post processing new mothod delte if not working
    def postprocess_sentence(self, sentence):
        if isinstance(sentence, int):
            sentence = [sentence]
        elif isinstance(sentence, list) and all(isinstance(i, int) for i in sentence):
            pass
        elif isinstance(sentence, list) and all(isinstance(i, list) for i in sentence):
            sentence = [item for sublist in sentence for item in sublist]
        else:
            sentence = sentence

        decoded_sentence = self.tokenizer.decode(sentence)
        #remove <start> and <end> tokens
        decoded_sentence = decoded_sentence.replace('start', '').replace('end', '').strip()
        
        #remove whitespace new line below
        decoded_sentence = re.sub(r'\s+', ' ', decoded_sentence).strip()

        #Fix splitup words 
        decoded_sentence = re.sub(r'\b(\w)\s(?=\w\b)', r'\1', decoded_sentence)
        # Specific fixes for common contractions
        decoded_sentence = re.sub(r"\b([Ii])\s+m\b", r"\1'm", decoded_sentence)          # i m -> I'm
        decoded_sentence = re.sub(r"\b([Yy]ou)\s+re\b", r"\1're", decoded_sentence)       # you re -> you're
        decoded_sentence = re.sub(r"\b([Ww]e)\s+re\b", r"\1're", decoded_sentence)         # we re -> we're
        decoded_sentence = re.sub(r"\b([Tt]hey)\s+re\b", r"\1're", decoded_sentence)       # they re -> they're
        decoded_sentence = re.sub(r"\b([Dd]on)\s+t\b", r"\1't", decoded_sentence)          # don t -> don't
        decoded_sentence = re.sub(r"\b([Cc]an)\s+t\b", r"\1't", decoded_sentence)          # can t -> can't
        decoded_sentence = re.sub(r"\b([Ww]on)\s+t\b", r"\1't", decoded_sentence)          # won t -> won't
        decoded_sentence = re.sub(r"\b([Dd]idn)\s+t\b", r"\1't", decoded_sentence)         # didn t -> didn't
        decoded_sentence = re.sub(r"\b([Dd]oesn)\s+t\b", r"\1't", decoded_sentence)         # doesn t -> doesn't
        decoded_sentence = re.sub(r"\b([Ss]houldn)\s+t\b", r"\1't", decoded_sentence)       # shouldn t -> shouldn't
        decoded_sentence = re.sub(r"\b([Cc]ouldn)\s+t\b", r"\1't", decoded_sentence)        # couldn t -> couldn't
        decoded_sentence = re.sub(r"\b([Ww]ouldn)\s+t\b", r"\1't", decoded_sentence)        # wouldn t -> wouldn't
        decoded_sentence = re.sub(r"\b([Ii])\s+ve\b", r"\1've", decoded_sentence)           # i ve -> i've
        decoded_sentence = re.sub(r"\b([Yy]ou)\s+ve\b", r"\1've", decoded_sentence)         # you ve -> you've
        decoded_sentence = re.sub(r"\b([Ww]e)\s+ve\b", r"\1've", decoded_sentence)           # we ve -> we've
        decoded_sentence = re.sub(r"\b([Tt]hey)\s+ve\b", r"\1've", decoded_sentence)         # they ve -> they've

        # Then, add a generic fix for cases like "what s" -> "what's"
        decoded_sentence = re.sub(r"\b(\w+)\s+s\b", r"\1's", decoded_sentence)
        # Optional: further fix split-up words if needed
        #decoded_sentence = re.sub(r'\b(\w)\s(?=\w\b)', r'\1', decoded_sentence)
        
        return decoded_sentence
    
    
    def calculate_confidencevalue_response(self, user_input, bot_response, fitness_score, candidates):
        """
        Calculate confidence based on fitness score, semantic similarity, intent match, and diversity.
        """
        #print(f"Inital Fitness Score: {fitness_score}")
        fitness_score = (fitness_score + 30) / 60
        #print(f"Final Adjusted Fitness Score: {fitness_score}")
        F = fitness_score
        
        
        #Semantic Similarity (S)
        user_embedding = self.semantic_model.encode(user_input, convert_to_tensor=True)
        bot_embedding = self.semantic_model.encode(bot_response, convert_to_tensor=True)
        S = util.pytorch_cos_sim(user_embedding, bot_embedding).item()

        #Response Diversity Factor (D)
        candidate_embeddings = [self.semantic_model.encode(c, convert_to_tensor=True) for c in candidates]
        avg_similarity = np.mean([util.pytorch_cos_sim(bot_embedding, c).item() for c in candidate_embeddings])
        D = 1 - avg_similarity  #Lower similarity = higher diversity

        #Compute final confidence score with weights
        #confidence = np.clip(0.3 * F + 0.3 * S + 0.4 * D, 0, 1) #more diversty in responses      this was old line 
        print(f"This is the F value: {F}")
        print(f"This is the S value: {S}")
        print(f"This is the D value: {D}")
        
        confidence = np.clip(0.5 * F + 0.3 * S + 0.2 * D, 0, 1)
        
        return confidence
    
    
    def generate_seq2seqresponse(self, input_sentence, num_candidates=5, temperature=0.6, top_k=30):
        #preprocess the input sentence and convert to a tensor
        input_sequence = self.preprocess_sentence(input_sentence)
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [input_sequence], maxlen=self.max_length, padding='post'
        )
        input_tensor = tf.convert_to_tensor(input_sequence)

        #prepare start/end tokens
        start_token_id = self.tokenizer.encode(self.start_token)[0]
        end_token_id = self.tokenizer.encode(self.end_token)[0]
        

        #collect multiple candidate responses from generation
        unique_candidates = set()
        all_candidates = []
        while len(all_candidates) < num_candidates:
            decoder_input = tf.expand_dims([start_token_id], 0)
            response_ids = []
            for _ in range(self.max_length):
                #pass through the model
                predictions = self.model([input_tensor, decoder_input])
                
                #apply temperature scaling
                predictions = predictions / temperature
                
                #convert to probabilities
                predicted_probabilities = tf.nn.softmax(predictions[:, -1, :], axis=-1).numpy()[0]
                
                #select top-k tokens
                top_k_indices = np.argsort(predicted_probabilities)[-top_k:]
                top_k_probs = predicted_probabilities[top_k_indices]
                
                #normalize the top-k probabilities
                top_k_probs /= np.sum(top_k_probs)
                
                #sample from the top-k tokens
                predicted_id = np.random.choice(top_k_indices, p=top_k_probs)
                
                #stop if we hit the end token
                if predicted_id == end_token_id:
                    break

                #append to response
                response_ids.append(predicted_id)
                
                #update decoder input
                decoder_input = tf.concat(
                    [decoder_input, tf.expand_dims([predicted_id], 0)], axis=-1
                )

            #postprocess token ids into a string
            candidate_text = self.postprocess_sentence(response_ids)
            if candidate_text not in unique_candidates:
                unique_candidates.add(candidate_text)
                all_candidates.append(candidate_text)
                print("New candidate:", candidate_text)
            else:
                print("Duplicate candidate detected. Regenerating...")
            
        #return the list of candidate strings
        return all_candidates


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
            for sent in doc.sents:
                # main root
                root = sent.root
                if root is not None:
                    clause_text = subtree_text(root)
                    if clause_text.strip():
                        clauses.append(clause_text)

                #optionally look for coordinated roots or conj
                #eg "He ran and he jumped"
                #we can gather other 'conj' heads that match the root or sentence boundary
                for token in sent:
                    if token.dep_ == "conj" and token.head == root:
                        conj_text = subtree_text(token)
                        if conj_text.strip():
                            clauses.append(conj_text)
            return clauses
        
        doc_a = nlp(parent_a)
        doc_b = nlp(parent_b)

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
        random.shuffle(child_clauses)  #optional shuffle

        #Duplicate Removal Only When Needed
        if len(set(child_clauses)) < len(child_clauses):  #check if duplicates exist
            child_clauses = list(dict.fromkeys(child_clauses))  #remove duplicates efficiently

        child_text = " ".join(child_clauses)
        print("Child text:", child_text)

        #optionally run a grammar correction pass would have to make a new method for this 
        #child_text = self.grammar_correct(child_text)

        return child_text


    #----------------------------------------------------------------------------------------



    def mutate_text_all_in_one(self, text: str) -> str:
        """
        A single function that performs advanced mutation:
        1) Parse text, extract clauses
        2) Randomly do one of:
            - remove a clause
            - reorder clauses
            - synonym-replace up to 2 content words in one random clause

        This function inlines:
        - Clause extraction
        - get similar word lookup
        - Clause-level mutation
        """

        # -------------- Inline Helpers ---------------

        def subtree_text(root_token):
            """
            Return the text of an entire subtree from 'root_token',
            including punctuation tokens in that subtree.
            """
            tokens = list(root_token.subtree)
            tokens = sorted(tokens, key=lambda t: t.i)
            return " ".join(t.text for t in tokens)

        def extract_clauses(doc):
            """
            Extract top-level clauses (root subtrees) from a spacy doc
            For each sentence we grab the subtree of the main root
            We also handle 'conj' heads that match the root for additional clauses
            """
            clauses = []
            for sent in doc.sents:
                root = sent.root
                if root is not None:
                    clause_text = subtree_text(root)
                    if clause_text.strip():
                        clauses.append(clause_text)

                # Look for conj tokens
                for token in sent:
                    if token.dep_ == "conj" and token.head == root:
                        conj_text = subtree_text(token)
                        if conj_text.strip():
                            clauses.append(conj_text)
            return clauses
        
        def get_similar_word(word, threshold=0.7, top_n=10):
            """
            Return a semantically similar word (if available) using word vectors.
            If no candidate meets the similarity threshold, return the original word.
            """
            #get the token from the word (using the first token of the doc)
            token = nlp(word)[0]
            if not token.has_vector:
                return word  #if no vector is available for this word, return it unchanged.
            
            candidates = []
            
            # Iterate over a filtered subset of the vocabulary.
            # This filters out words that are not lower-case not alphabetic or are very rare.
            for lex in nlp.vocab:
                if (lex.has_vector and lex.is_lower and lex.is_alpha and 
                    lex.prob > -15 and lex.text != word):
                    sim = token.similarity(lex)
                    if sim >= threshold:
                        candidates.append((lex.text, sim))
            
            #sort candidates by similarity (highest first) and take the top_n
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]
            
            if candidates:
                #randomly choose one of the top candidates
                return random.choice([w for w, sim in candidates])
            
            return word

        # -------------- Start of Actual Mutation Logic ---------------

        doc = nlp(text)
        clauses = extract_clauses(doc)
        if not clauses:
            return text  # nothing to mutate

        # We'll pick one of three mutation types
        mutation_type = random.choice(["remove_clause", "reorder", "synonym_replace"])
        
        mutated_clauses = clauses[:]  # Initialize with the original clauses

        if mutation_type == "remove_clause" and len(clauses) > 1:
            remove_idx = random.randint(0, len(clauses) - 1)
            mutated_clauses.pop(remove_idx)  
            print("mutated clauses before mutation type remove clause", mutated_clauses)
            mutated_text = " ".join(mutated_clauses)

        # 2) reorder
        elif mutation_type == "reorder" and len(clauses) > 1:
            #old line for below new line random.shuffle(clauses)
            random.shuffle(mutated_clauses)
            print("mutated clauses before mutation type remove clause", mutated_clauses)
            mutated_text = " ".join(mutated_clauses)

        # 3) synonym_replace
        else:
            # pick one random clause to synonym-replace up to 2 content words
            clause_idx = random.randint(0, len(clauses) - 1)
            chosen_clause = clauses[clause_idx]

            # parse the chosen clause
            clause_doc = nlp(chosen_clause)
            tokens = [t for t in clause_doc]

            # find content tokens
            content_indices = [i for i, t in enumerate(tokens)
                            if t.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
            if not content_indices:
                # if no content words, we can't do synonyms; fallback to original
                #old line for below new line mutated_clauses = clauses
                mutated_clauses = clauses[:]
            else:
                # pick how many replacements we do (1 or 2)
                replace_count = random.randint(1, min(2, len(content_indices)))
                indices_to_replace = random.sample(content_indices, replace_count)

                mutated_tokens = tokens[:]
                for idx_replace in indices_to_replace:
                    old_token = tokens[idx_replace]
                    new_word = get_similar_word(old_token.text, threshold=0.7, top_n=10)
                    if new_word != old_token.text:
                        print(f"Replacing '{old_token.text}' with '{new_word}'")
                        mutated_tokens[idx_replace] = new_word
                        #         mutated_tokens[idx_replace] = nlp(new_word)[0].text  # Ensures correct formatting

                        
                # reconstruct mutated clause
                mutated_clause_text = " ".join(t if isinstance(t, str) else t.text
                                            for t in mutated_tokens)
                mutated_clauses = clauses[:clause_idx] + [mutated_clause_text] + clauses[clause_idx+1:]

            

            if len(set(mutated_clauses)) < len(mutated_clauses):
                print("Duplicates detected and removed:", set(mutated_clauses))
                mutated_clauses = list(dict.fromkeys(mutated_clauses))
            
            print("Mutated clauses:", mutated_clauses)
            
            mutated_text = " ".join(mutated_clauses)
            print("Mutated text:", mutated_text)
        

        return mutated_text

    def get_fitness_score(self, user_input, bot_response):
        # This version does not rely on a running self.reward_score.
        reward = 0

        # (1) Do the same checks you do in evaluate_response
        user_doc = self.nlp(user_input)
        bot_doc = self.nlp(bot_response)
        user_intent = self.intent_classifier.predict_intent(user_input)
        bot_intent = self.intent_classifier.predict_intent(bot_response)

        relevance, similarity = self.sentiment_analyzer.check_relevance(user_doc, bot_doc, user_input, bot_response)
        sentiment_score = self.sentiment_analyzer.analyze_sentiment(user_input, bot_response)
        intent_match = (user_intent == bot_intent)

        # (2) Adjust 'reward' but not self.reward_score
        if relevance:
            reward += 10
        if similarity > 0.5:
            reward += 5
        if sentiment_score < 0.1:
            reward += 5
        if intent_match:
            reward += 10

        if not relevance:
            reward -= 10
        if similarity < 0.3:
            reward -= 5
        if sentiment_score > 0.5:
            reward -= 5
        if not intent_match:
            reward -= 10

        return reward
    
    def ga_rerank_candidates(self, user_input, candidates,
                            pop_size=10, generations=3,
                            crossover_rate=0.5, mutation_rate=0.3): #mutation rate was 0.1 orginally generations was 3
        """
        1) Start with 'candidates' as the initial population.
        2) Evaluate them with 'reward_system' as the fitness.
        3) Evolve for 'generations' times.
        4) Return the best final string.
        """
        # ensure population has at least pop_size
        candidates = list(set(candidates))  # Remove duplicates
        population = candidates[:pop_size]
        while len(population) < pop_size:
            population.append(random.choice(candidates))

        def fitness_func(candidate_text):
            #used to be reward_system.get_fitness_score in the below line
            return self.get_fitness_score(user_input, candidate_text)

        # Evaluate initial population
        #scores = [fitness_func(c) for c in population] # old line new below
        scores = np.array([fitness_func(c) for c in population]) #new optimized line with nummpy less memory useage delete line and use one above if issue

        for gen in range(generations):
            # Selection: pick top half as parents
            ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            parents = [p for p, s in ranked[:pop_size // 2]]

            # Make new population
            new_population = []
            while len(new_population) < pop_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                child = parent1
                if random.random() < crossover_rate:
                    child = self.crossover_text_advanced(parent1, parent2)

                if random.random() < mutation_rate:
                    child = self.mutate_text_all_in_one(child)

                new_population.append(child)

            population = new_population
            #scores = [fitness_func(c) for c in population] # old line new below
            scores = np.array([fitness_func(c) for c in population]) #new optimized line with nummpy less memory useage delete line and use one above if issue

        # Finally, pick the best
        best_idx = max(range(len(population)), key=lambda i: scores[i])
        best_candidate = population[best_idx] 
        print("best canidate GA method return:", best_candidate)
        best_score = scores[best_idx]
        
        #trying to do postprocessing below diffrent order new line below

        return best_candidate, best_score
        


    def modeltrain(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for (batch, (encoder_inputs, decoder_inputs)) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    logits = self.model([encoder_inputs, decoder_inputs])
                    loss = self.compute_loss(decoder_inputs[:, 1:], logits[:, :-1, :])
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                total_loss += loss
            print('Epoch {}, Loss {:.4f}'.format(epoch + 1, total_loss))

    def compute_loss(self, labels, logits):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        mask = tf.cast(mask, dtype=tf.float32)
        loss_value = loss(labels, logits, sample_weight=mask)
        return loss_value

    def save_model(self):
        self.model.save_weights(script_dir / 'model_weights.weights.h5')

        #if issue use the save weights line
        #self.model.save('/Users//Desktop/hello/model_weights', save_format="tf")
            
    def load_model(self):
        if os.path.exists(script_dir / 'model_weights.weights.h5'):
            #recreate the model with the known vocab size
            self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)

            #dummy call to initialize the variables
            dummy_input = [tf.zeros((1, 1)), tf.zeros((1, 1))]
            self.model(dummy_input)

            #Load the weights
            self.model.load_weights(script_dir / 'model_weights.weights.h5')
                