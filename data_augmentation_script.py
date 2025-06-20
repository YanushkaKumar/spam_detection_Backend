import pandas as pd
import numpy as np
import re
import random
from sklearn.utils import resample
from collections import Counter
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

class DataAugmentation:
    def __init__(self):
        self.spam_patterns = [
            r'\bfree\b', r'\bwin\b', r'\bmoney\b', r'\bcash\b', r'\bprize\b',
            r'\burgent\b', r'\blimited\b', r'\boffer\b', r'\bdeal\b',
            r'\bclick here\b', r'\bcall now\b', r'\bact now\b'
        ]
        
    def synonym_replacement(self, text, n=1):
        """Replace words with synonyms using nlpaug"""
        try:
            aug = naw.SynonymAug(aug_src='wordnet')
            augmented_text = aug.augment(text)
            return augmented_text if isinstance(augmented_text, str) else text
        except:
            return text
    
    def random_insertion(self, text, n=1):
        """Insert random spam-related words"""
        spam_words = ['free', 'win', 'cash', 'urgent', 'limited', 'offer', 'deal']
        words = text.split()
        
        for _ in range(n):
            random_word = random.choice(spam_words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def random_swap(self, text, n=1):
        """Randomly swap words in the text"""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text, p=0.1):
        """Randomly delete words from the text"""
        words = text.split()
        if len(words) == 1:
            return text
            
        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        
        return ' '.join(new_words) if new_words else text
    
    def character_augmentation(self, text):
        """Apply character-level augmentation"""
        try:
            # Character substitution
            aug = nac.KeyboardAug()
            augmented_text = aug.augment(text)
            return augmented_text if isinstance(augmented_text, str) else text
        except:
            return text
    
    def create_spam_variations(self, text):
        """Create spam variations by adding common spam patterns"""
        variations = []
        
        # Add urgency words
        urgency_words = ['URGENT!!!', 'ACT NOW!', 'LIMITED TIME!', 'HURRY!']
        for word in urgency_words:
            variations