from sentence_transformers import SentenceTransformer, util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nlp.intent import IntentClassifier
from config import script_dir, nlp



class DynamicRewardSystem:
    def __init__(self):
        self.nlp = nlp
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        #self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #maybe remove these
        #self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2') #maybe remove these
        self.intent_classifier = IntentClassifier()  
        self.reward_score = 0

    def evaluate_response(self, user_input, bot_response):
        
        self.reward_score = 0  # Reset reward score for each evaluation this is a massive test to ensure new normalziation between -30 and 30 (0 and 1) works
        
        # Process the texts
        user_doc = self.nlp(user_input)
        bot_doc = self.nlp(bot_response)
        user_intent = self.intent_classifier.predict_intent(user_input)
        bot_intent = self.intent_classifier.predict_intent(bot_response)

        #Semantic and Contextual Analysis
        relevance, similarity = self.check_relevance(user_doc, bot_doc, user_input, bot_response)
        intent_match = user_intent == bot_intent  #intents match?

        #sentiment analysis
        sentiment_score = self.analyze_sentiment(user_input, bot_response)

        #update Reward
        self.update_reward(relevance, similarity, sentiment_score, intent_match)

        print(f"\nUpdated Reward Score: {self.reward_score}")
        return self.reward_score
    
    def check_relevance(self, user_doc, bot_doc, user_input, bot_response):
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        bot_embedding = self.model.encode(bot_response, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(user_embedding, bot_embedding).item()

        contextual_match = len(set([chunk.text for chunk in user_doc.noun_chunks]) & set([chunk.text for chunk in bot_doc.noun_chunks])) > 0
        dependency_match = any(token.dep_ == bot_token.dep_ for token in user_doc for bot_token in bot_doc)

        relevance = similarity > 0.3 and (contextual_match or dependency_match)
        return relevance, similarity

    def analyze_sentiment(self, user_input, bot_response):
        user_sentiment = self.sentiment_analyzer.polarity_scores(user_input)['compound']
        bot_sentiment = self.sentiment_analyzer.polarity_scores(bot_response)['compound']
        return abs(user_sentiment - bot_sentiment)
    
    def update_reward(self, relevance, similarity, sentiment_score, intent_match):
        #positive rewards
        if relevance:
            self.reward_score += 10  #increase reward if the response is relevant
        if similarity > 0.5:
            self.reward_score += 5  #additional reward for high similarity
        if sentiment_score < 0.1:
            self.reward_score += 5  #reward alignment in sentiment
        if intent_match:
            self.reward_score += 10  #reward for matching intents

        #penalties
        if not relevance:
            self.reward_score -= 10  #penalty for irrelevant response
        if similarity < 0.3:
            self.reward_score -= 5  #penalty for low similarity
        if sentiment_score > 0.5:
            self.reward_score -= 5  #penalty for poor sentiment alignment
        if not intent_match:
            self.reward_score -= 10  #penalty for mismatched intents


    def get_total_reward(self):
        return self.reward_score