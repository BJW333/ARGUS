from sentence_transformers import SentenceTransformer, util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nlp.intent import IntentClassifier #not used currently due to intent matching being unreliable 
from config import script_dir, nlp
import language_tool_python
import re
from metrics.logging import log_debug, log_metrics  

class DynamicRewardSystem:
    def __init__(self):
        self.nlp = nlp
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        #self.intent_classifier = IntentClassifier()  #not used anyone because intent matching unreliable
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.reward_score = 0

    def check_relevance(self, user_doc, bot_doc, user_input, bot_response):
        user_embedding = self.semantic_model.encode(user_input, convert_to_tensor=True, show_progress_bar=False)
        bot_embedding = self.semantic_model.encode(bot_response, convert_to_tensor=True, show_progress_bar=False)
        similarity = util.pytorch_cos_sim(user_embedding, bot_embedding).item()

        contextual_match = len(set([chunk.text for chunk in user_doc.noun_chunks]) & set([chunk.text for chunk in bot_doc.noun_chunks])) > 0
        dependency_match = any(token.dep_ == bot_token.dep_ for token in user_doc for bot_token in bot_doc)

        relevance = similarity > 0.3 and (contextual_match or dependency_match)
        return relevance, similarity

    def analyze_sentiment(self, user_input, bot_response):
        user_sentiment = self.sentiment_analyzer.polarity_scores(user_input)['compound']
        bot_sentiment = self.sentiment_analyzer.polarity_scores(bot_response)['compound']
        return abs(user_sentiment - bot_sentiment)
    
    def evaluate_response(self, user_input, bot_response):
        self.reward_score = 0  # Reset reward score for each evaluation this is a massive test to ensure new normalziation between -30 and 30 (0 and 1) works
        
        # Process the texts
        user_doc = self.nlp(user_input)
        bot_doc = self.nlp(bot_response)
        #user_intent = self.intent_classifier.predict_intent(user_input)
        #bot_intent = self.intent_classifier.predict_intent(bot_response)

        #Semantic and Contextual Analysis # 1. Relevance # 2. Semantic Similarity
        relevance, similarity = self.check_relevance(user_doc, bot_doc, user_input, bot_response)
        #intent_match = user_intent == bot_intent  #intents match?

        #sentiment analysis # 3. Sentiment Alignment
        sentiment_score = self.analyze_sentiment(user_input, bot_response)

        # 4. Clarity
        word_count = len(bot_response.split())

        # 5. Grammar
        grammar_issues = len(self.grammar_tool.check(bot_response))
        
        vague_words = ['maybe', 'probably', 'possibly', 'could', 'might', 'i think']
        vague_count = sum(len(re.findall(rf"\b{w}\b", bot_response.lower())) for w in vague_words)
        
        direct_answer = bool(re.search(r"\b(refers to|means|defined as|consists of)\b", bot_response.lower()))

        novelty = 1 - similarity
        
        #update Reward
        #self.update_reward(relevance, similarity, sentiment_score, intent_match)
        self.update_reward(relevance, similarity, sentiment_score, direct_answer, word_count, grammar_issues, vague_count, novelty)

        log_debug(f"\nUpdated Reward Score: {self.reward_score}") #debugging line
        
        return self.reward_score
    
    #def update_reward(self, relevance, similarity, sentiment_score, intent_match): #not used currently due to intent matching being unreliable
    def update_reward(self, relevance, similarity, sentiment_score, direct_answer, word_count, grammar_issues, vague_count, novelty):
        #if intent_match: # Intent matches
        #    self.reward_score += 10
        #if not intent_match: # Intent does not match
        #    self.reward_score -= 10

        #intent matching currently intent matching is unreliable until a new system is designed or trained 
        
        
        # Relevance
        if relevance:
            self.reward_score += 10 # Strong relevance
        else:
            self.reward_score -= 10 # Weak relevance
            
        # Semantic Similarity    
        if similarity > 0.7:
            self.reward_score += 5  # Very strong alignment
        elif similarity > 0.5:
            self.reward_score += 3  # Good alignment
        elif similarity > 0.3:
            self.reward_score += 1  # Somewhat related, give a little credit
        else:
            self.reward_score -= 5  # Weak or unrelated response
        
        # Sentiment Alignment
        if sentiment_score < 0.3:
            self.reward_score += 5
        elif sentiment_score < 0.6:
            self.reward_score += 2
        else:
            self.reward_score -= 5
            
        # Clarity   
        if 6 <= word_count <= 40:
            self.reward_score += 5
        elif word_count < 6:
            self.reward_score -= 5
        else:
            self.reward_score -= 2    
            
        # Grammar   
        if grammar_issues == 0:
            self.reward_score += 5
        elif grammar_issues <= 2:
            self.reward_score += 2
        else:
            self.reward_score -= 5    
            
        # Specificity    
        if vague_count == 0:
            self.reward_score += 5
        elif vague_count <= 2:
            self.reward_score += 2
        else:
            self.reward_score -= 5    
            
        # Direct Answer 
        if direct_answer:  # Direct answer is present
            self.reward_score += 5
        else:              # Direct answer is not present
            self.reward_score -= 5
        
        # Novelty
        if novelty > 0.4:
            self.reward_score += 5
        elif novelty < 0.1:
            self.reward_score -= 5
        
    def get_total_reward(self):
        return self.reward_score
    
    