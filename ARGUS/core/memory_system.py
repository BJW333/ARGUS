import json
import os 
import re
import time
import numpy as np 
from filelock import FileLock
from typing import Dict, List, Union
from sentence_transformers import SentenceTransformer
from core.input_bus import print_to_gui
from speech.speak import speak
from speech.listen import generalvoiceinput

class MemoryManager():
    """
    A three-tier memory system:
    1. Personality Memory  - Fixed traits and user preferences.
    2. Long-Term Memory    - Important facts and tasks, retrieved via semantic search.
    3. Short-Term Memory   - Last few conversation turns for immediate context.
    """
    #CONFIG 
    SHORT_TERM_LIMIT = 10        #number of turns to store in short-term memory
    IMPORTANCE_THRESHOLD = 4     #score needed to store something in long-term memory
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  #sentence embedding model (384-dim)

    #regex patterns for detecting important info.  #this needs major overhaul its not good enough
    IDENTITY_RE  = re.compile(r"(?:i am|i'm|call me|my name is)\s+([\w\s'-]{2,50})", re.I)
    REMEMBER_RE  = re.compile(r"\b(remember|don't forget)\b", re.I)

    #stop words to ignore in preferences, dislikes, skills, and goals
    STOPWORDS = {"to","and","the","a","an","of","in","on"}

    #patterns for detecting preferences, dislikes, skills, and goals
    PREFERENCE_RE = re.compile(
        r"\bi\s+(?:really\s+)?(?:like|love|enjoy|prefer)\s+(?:to\s+)?(?P<preference>(?!to\b)[a-z][a-z\s'-]{2,40})\b",
        re.I,
    )
    DISLIKE_RE = re.compile(
        r"\bi\s+(?:don't like|hate|dislike|can't stand)\s+(?:to\s+)?(?P<dislike>(?!to\b)[a-z][a-z\s'-]{2,40})\b",
        re.I,
    )
    SKILL_RE = re.compile(
        r"\bi\s+(?:can|know how to|am good at|have experience with)\s+(?:to\s+)?(?P<skill>(?!to\b)[a-z][a-z\s'-]{2,40})\b",
        re.I,
    )
    GOAL_RE = re.compile(
        r"\bi\s+(?:want|hope|plan|aim|intend|need)\s+(?:to\s+)?(?P<goal>(?!to\b)[a-z][a-z\s'-]{2,40})\b",
        re.I,
    )

    PROJECT_KEYWORDS = {
        "project","task","work","assignment",
        "build","create","develop","design",
        "implement","deploy","launch",
        "optimize","improve","refactor",
        "building","creating","developing","designing",
        "implementing","deploying","launching",
    }
    TASK_NEAR_I_RE = re.compile(
        r"\bI\b(?:\W+\w+){0,4}\W+(?:" + "|".join(re.escape(w) for w in sorted(PROJECT_KEYWORDS)) + r")\b",
        re.I,
    )   

    # Use double braces for regex quantifiers
    FORGET_TEMPL = (
        r"(?:forget|remove|i don't want|i no longer want|stop|change)\s+"
        r"(?:that\s+)?i\s+{kw}\s+(?P<value>[a-zA-Z\s'-]{{2,40}})"
    )

    FORGET_REGEX = {
        "preferences": re.compile(FORGET_TEMPL.format(kw="like"), re.I),
        "dislikes":    re.compile(FORGET_TEMPL.format(kw="(?:don't like|hate|dislike|can't stand)"), re.I),
        "skills":      re.compile(FORGET_TEMPL.format(kw="(?:can|know how to|am good at|have experience with)"), re.I),
        "goals":       re.compile(FORGET_TEMPL.format(kw="(?:want|hope|plan|aim|intend|need)"), re.I),
    }
    
    
    def __init__(self, memory_path: str = "memory") -> None:
        os.makedirs(memory_path, exist_ok=True)
        
        #personality memory json path
        self.personality_file = os.path.join(memory_path, "personality.json")
        
        #long term memory json path
        self.long_term_file = os.path.join(memory_path, "long_term.json")
        
        #short term memory json path
        self.short_term_file = os.path.join(memory_path, "short_term.json")
    
        #load memories from disk or create defaults
        self.personality_memory = self._load_json(self.personality_file, default={})
        self.long_term_memory = self._load_json(self.long_term_file, default=[])
        self.short_term_memory = self._load_json(self.short_term_file, default=[])
        
        self._cleanup_personality()  #clean up personality memory on init
        
    @classmethod
    def _get_embedder(cls):
        if not hasattr(cls, "_cached_embedder"):
            cls._cached_embedder = SentenceTransformer(cls.EMBEDDING_MODEL)
        return cls._cached_embedder    
    
    def _cleanup_personality(self) -> None:
        changed = False
        for key in ("preferences", "dislikes", "skills", "goals"):
            vals = self.personality_memory.get(key)
            if not isinstance(vals, list):
                continue
            cleaned, seen = [], set()
            for v in vals:
                nv = self._normalize_value(str(v))
                if nv and nv not in self.STOPWORDS and len(nv) >= 3 and nv not in seen:
                    seen.add(nv); cleaned.append(nv)
            if cleaned != vals:
                self.personality_memory[key] = cleaned
                changed = True
        if changed:
            self._save_json(self.personality_file, self.personality_memory)
            
    def is_task_sentence(self, text: str) -> bool:
        return bool(self.TASK_NEAR_I_RE.search(text))
            
    def update_memory_from_conversation(self, conversation_history: List[Dict]):
        """Main entry point: updates all memory from full conversation_history."""
        dirty_st = False
        for turn in conversation_history:
            user_msg = turn["user_input"]
            bot_msg  = turn["bot_response"]
            reward   = turn.get("reward", 0.0)

            ##append to short term memory
            self.short_term_memory.append({
                "user_input": user_msg,
                "bot_response": bot_msg,
                "reward": reward,
                "ts": time.time(),
            })
            self.short_term_memory = self.short_term_memory[-self.SHORT_TERM_LIMIT:]
            dirty_st = True

            #update long-term/personality memory if important
            self.add_turn(user_msg, bot_msg, reward)

        if dirty_st:
            self._save_json(self.short_term_file, self.short_term_memory)
            
    def add_turn(self, user_msg: str, bot_msg: str, reward: float = 0.0) -> None:
        """
        Route a new conversation turn into personality, long-term, or short-term memory.
        """
        flags = self.detect_flags_for_importance_scoring(user_msg)
        
        novelty, emb = self.compute_novelty(user_msg + " " + bot_msg, return_vec=True)
        
        importance = self.importance_score_weighting(flags, reward, novelty)

        if flags["identity"]:
            self._update_personality(user_msg)

        self._update_personality_traits(user_msg) 
        
        self.forget_or_update_traits(user_msg)
        
        if importance >= self.IMPORTANCE_THRESHOLD:
            self._save_long_term(user_msg, bot_msg, emb)
            
            
    def detect_flags_for_importance_scoring(self, text: str) -> Dict[str, bool]:
        return {
            "identity": bool(self.IDENTITY_RE.search(text)),
            "explicit": bool(self.REMEMBER_RE.search(text)),
            "task": self.is_task_sentence(text),
        }

    def importance_score_weighting(self, f: Dict[str, bool], reward: float, novelty: float) -> float:
        """Weighted sum of flags + reward – duplicate penalty."""
        return (
            3 * f["explicit"]
            + 2 * f["identity"]
            + 2 * f["task"]
            + reward / 10
            + novelty          # reward, don’t penalise
        )
    
    def compute_novelty(self, text: str, return_vec: bool = False):
        vec = self._embed(text)
        if not self.long_term_memory:
            return (1.0, vec) if return_vec else 1.0

        embs = [np.array(e["embedding"]) for e in self.long_term_memory if e.get("embedding")]
        best_sim = max(
            (self._cosine_similarity(vec, e) for e in embs),
            default=0.0
        )
        # Normalize novelty to [0, 1] range
        novelty = 1.0 - min(1.0, max(0.0, best_sim))
        
        return (novelty, vec) if return_vec else novelty    
    
    def _update_personality(self, user_msg: str):
        m = self.IDENTITY_RE.search(user_msg)
        if m:
            name = m.group(1).strip(" .,'-")
            if name:
                self.personality_memory["name"] = name
                self._save_json(self.personality_file, self.personality_memory)
            
    def _normalize_value(self, v: str) -> str:
        v = " ".join(v.split()).strip(" -'").lower()
        return v

    def _update_personality_traits(self, user_msg: str):
        patterns = {
            "preferences": (self.PREFERENCE_RE, "preference"),
            "dislikes":    (self.DISLIKE_RE, "dislike"),
            "skills":      (self.SKILL_RE, "skill"),
            "goals":       (self.GOAL_RE, "goal")
        }
        changed = False
        for category, (rx, group_name) in patterns.items():
            for m in rx.finditer(user_msg):
                value = self._normalize_value(m.group(group_name))
                if not value or value in self.STOPWORDS or len(value) < 3:
                    continue
                self.personality_memory.setdefault(category, [])
                if value not in self.personality_memory[category]:
                    self.personality_memory[category].append(value)
                    changed = True
        if changed:
            self._save_json(self.personality_file, self.personality_memory) 
    
    def forget_or_update_traits(self, user_msg: str):
        lowered = user_msg.lower()
        changed = False

        # Forget name
        if "forget my name" in lowered or re.search(r"\b(forget|reset) (my )?name\b", lowered):
            if "name" in self.personality_memory:
                del self.personality_memory["name"]
                changed = True
                
        elif "need to change my name that is saved" in lowered or re.search(r"\b(change|alter) (my )?name\b", lowered):
            if "name" in self.personality_memory:
                print_to_gui("Do you really want to change your name in the system?: ")
                print_to_gui("Yes or No")
                speak("Do you really want to change your name in the system? Answer Yes or No")
                changenameYorN = generalvoiceinput()
                if "yes" in changenameYorN:
                    print_to_gui("What do you want your name changed to?: ")
                    speak("What do you want your name changed to?")
                    newname = generalvoiceinput()
                    self.personality_memory["name"] = newname
                    print_to_gui(f"This is the new updated name: {newname}")
                    speak(f"I will now call you {newname}")
                else:
                    speak("Ok I wont change your name in the system")


        # Forget traits (all matches)
        for category, regex in self.FORGET_REGEX.items():
            for m in regex.finditer(lowered):
                value = self._normalize_value(m.group("value"))
                lst = self.personality_memory.get(category, [])
                if value in lst:
                    lst.remove(value)
                    changed = True

        if changed:
            self._save_json(self.personality_file, self.personality_memory)     
         
    def personalitymemory(self):
        """    
        Personality memory: Fixed traits and user preferences.
        """
        return self.personality_memory

    def retrieved_memories(self, query: str, top_k: int = 5, min_similarity: float = 0.7) -> List[Dict[str, str]]:
        """    
        Long-term memory: Search-based relevance from older context
        """
        if not self.long_term_memory:
            return []
        query_vec = self._embed(query)
        scored = []
        for entry in self.long_term_memory:
            emb = entry.get("embedding")
            if emb is None:
                continue
            sim = self._cosine_similarity(query_vec, np.array(emb))
            if sim >= min_similarity:
                scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]
    
    def recent_convo(self):
        """
        Short-term memory: Last N turns (dialogue)
        """
        return self.short_term_memory[-self.SHORT_TERM_LIMIT:]
   
    def _save_long_term(self, user_msg: str, bot_msg: str, new_embedding: np.ndarray | None = None) -> None:
        """
        Save a new long-term memory entry if not a duplicate.
        """
        self.decay_old_memories(max_age_days=15) #auto prune old memorys before saving new memorys

        if new_embedding is None:
            new_embedding = self._embed(user_msg + " " + bot_msg)

        for entry in self.long_term_memory:
            emb = entry.get("embedding")
            if not emb:
                continue
            sim = self._cosine_similarity(new_embedding, np.array(emb))
            if sim > 0.85: #similarity threshold
                print("Skipping duplicate memory entry.")
                return

        self.long_term_memory.append({
            "user": user_msg,
            "bot": bot_msg,
            "embedding": new_embedding.tolist(),
            "timestamp": time.time()  #time stamp enables pruneing and decay of old memories
        })
        self._save_json(self.long_term_file, self.long_term_memory)
        
    def decay_old_memories(self, max_age_days: int = 15) -> None:
        """Remove long-term memories older than max_age_days."""
        current_time = time.time()
        max_age_seconds = max_age_days * 86400  # 60 * 60 * 24
        before = len(self.long_term_memory)
        self.long_term_memory = [
            entry for entry in self.long_term_memory
            if current_time - entry.get("timestamp", current_time) < max_age_seconds
        ]
        after = len(self.long_term_memory)
        if before != after:
            print(f"Pruned {before - after} stale long-term memory entries.")
            self._save_json(self.long_term_file, self.long_term_memory)
    
    def get_all_memories(self, current_input: str) -> Dict[str, Union[dict, List[Dict]]]:
        lt = self.retrieved_memories(current_input) # semantically relevant
        lt_clean = [
            {"user": e.get("user"), "bot": e.get("bot"), "timestamp": e.get("timestamp")}
            for e in lt
        ]
        return {
            "personality": self.personalitymemory(),
            "short_term": self.recent_convo(), # returns last N turns
            "long_term": lt_clean,   # ← no embeddings in prompt
        }    
        
    #below are essiential methods for memory management
    def _embed(self, text: str) -> np.ndarray:
        """Embed text using the cached SentenceTransformer instance."""
        return self._get_embedder().encode(text, show_progress_bar=False) # turned off tqdm
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def _load_json(path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # backup the corrupt file and return default
            try:
                os.replace(path, path + f".corrupt.{int(time.time())}")
            except Exception:
                pass
            return default

        # Enforce embedding dimension dynamically for long_term
        if path.endswith("long_term.json"):
            try:
                dim = MemoryManager._get_embedder().get_sentence_embedding_dimension()
            except Exception:
                dim = None
            if dim:
                data = [e for e in data if len(e.get("embedding", [])) == dim]
        return data

    @staticmethod
    def _save_json(path: str, data) -> None:
        tmp = path + ".tmp"
        lock = FileLock(path + ".lock")
        with lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # atomic on POSIX/modern Windows