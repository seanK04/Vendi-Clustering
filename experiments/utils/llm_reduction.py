"""LLM-assisted topic reduction baseline (Protocol 2).

Ported from the paper's fork. Uses only public BERTopic API, so it takes any
fitted model.
"""
import json
import re
import time
from typing import Any, List
import pandas as pd

class LLMReducer:
    def __init__(self, llm_client, model_name="gemma2:9b", top_keywords: int = 5):
        self.llm_client = llm_client
        self.model_name = model_name
        self.top_keywords = top_keywords

        # System prompt is identical in both representations (Janssens et al., Table 2)
        self.system_prompt = "You are a helpful assistant tasked with identifying overlapping topics."

        if top_keywords >= 10:
            # Paper Table 2: 10-word representation prompt (used for OpenAI flow)
            self.instruction = (
                "Below I will provide a numbered list of topics. "
                "Each line represents the topic words of a topic. "
                "Return the index of the two most similar topics in Python list format.\n"
                "[TOPICS]\n"
                "Return only the Python list and nothing more!"
            )

            self.demo_prompt = (
                "Below I will provide a numbered list of topic words. "
                "Each line represents the topic words of a topic. "
                "Return the index of the two most similar topics in Python list format.\n\n"
                "0: astronaut, planet, galaxy, NASA, space, orbit, cosmos\n"
                "1: apple, fruit, red, sweet, tree, juicy, pie, snack, healthy, crunchy\n"
                "2: computer, laptop, technology, device, screen, keyboard, mouse, software, internet, code\n"
                "3: orange, fruit, citrus, vitamin, juicy, peel, tangy, sweet, snack, healthy\n"
                "4: car, vehicle, engine, road, drive, fuel, tire, speed, traffic, transportation\n"
                "5: book, library, read, pages, author, fiction, chapter, novel\n\n"
                "Return only the Python list and nothing more!"
            )
        else:
            # Original 5-word representation prompt (used for Gemma flow)
            self.instruction = (
                "Below I will provide a numbered list of topics. "
                "Each line represents the topic words of a topic. "
                "Return the index of the two most similar topics in Python list format.\n"

                "Return only the Python list and nothing more!"
            )

            self.demo_prompt = (
                "Below I will provide a numbered list of topic words. "
                "Each line represents the topic words of a topic. "
                "Return the index of the two most similar topics in Python list format.\n\n"
                "0: astronaut, rocket, planet, galaxy, NASA\n"
                "1: apple, fruit, red, sweet, tree\n"
                "2: computer, laptop, technology, device, screen\n"
                "3: orange, fruit, citrus, vitamin, juicy\n"
                "4: car, vehicle, engine, road, drive\n"
                "5: book, library, read, pages, author, fiction\n\n"
                "Return only the Python list and nothing more!"
            )

        self.demo_answer = "[1, 3]"

    def _format_topic_list(self, topic_info: pd.DataFrame) -> str:
        """
        Formats BERTopic info into the exact text representation for the LLM.
        Matches the paper: '0: word1, word2, word3, word4, word5'
        """
        formatted_lines = []
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1: 
                continue
            
            # Get the keywords
            words = row['Representation']
            
            # Slice to the configured number of top keywords (5 for Gemma flow, 10 for OpenAI flow)
            if isinstance(words, list):
                top_words = words[:self.top_keywords]
                words_str = ", ".join(top_words)
            else:
                # Fallback if it's already a string (though unusual for BERTopic)
                words_str = str(words)
                
            formatted_lines.append(f"{topic_id}: {words_str}")
            
        return "\n".join(formatted_lines)

    def get_merge_pair(self, topic_info: pd.DataFrame) -> List[int]:
            """Calls the LLM to get the next two IDs to merge."""
            current_topics_formatted = self._format_topic_list(topic_info)
            
            # 1. Fill the [TOPICS] placeholder in the instruction
            task_with_topics = self.instruction.replace("[TOPICS]", current_topics_formatted)
            
            # 2. Construct the main prompt (few-shot example + current task)
            # We don't include "System:" here because the client handles that role.
            main_prompt = (
                f"Example Task:\n{self.demo_prompt}\n"
                f"Example Answer: {self.demo_answer}\n\n"
                f"Current Task:\n{task_with_topics}\n"
                f"Answer:"
            )

            # 3. Call the updated client with two arguments
            # This matches your client's: generate(self, system_prompt: str, main_prompt: str)
            response = self.llm_client.generate(
                system_prompt=self.system_prompt, 
                main_prompt=main_prompt
            )

            
            
            # Robust parsing for extracting [ID1, ID2]
            try:
                match = re.search(r'\[\s*#?(\d+)\s*,\s*#?(\d+)\s*\]', response)
                if match:
                    return [int(match.group(1)), int(match.group(2))]
            except Exception as e:
                print(f"Parsing error: {e}. Response was: {response}")
            
            return []

    def reduce(self, topic_model: Any, docs: List[str], target_k: int):
        """Iterative LLM-based reduction with robust state tracking."""
        
        # Track metrics
        merge_history = []
        api_calls = 0
        total_llm_time = 0.0
        
        while True:
            current_topics = set(topic_model.topics_) - {-1}
            current_k = len(current_topics)
            
            if current_k <= target_k:
                break
            
            # Get topic info (use more context as k decreases)
            context_size = current_k
            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info['Topic'] != -1].head(context_size)
            
            # Call LLM
            start_time = time.time()
            pair = self.get_merge_pair(topic_info)
            llm_time = time.time() - start_time
            total_llm_time += llm_time
            api_calls += 1
            
            # Validate pair
            if len(pair) != 2:
                print(f"  Invalid pair format: {pair}")
                break
                
            if not all(p in current_topics for p in pair):
                print(f"  Invalid topics: {pair} not in current topics")
                break
            
            # Perform merge
            print(f"  [{api_calls}] Merging {pair[0]} ← {pair[1]} (LLM: {llm_time:.2f}s)")
            merge_start = time.time()
            topic_model.merge_topics(docs, pair)
            merge_time = time.time() - merge_start
            
            # Verify merge succeeded
            new_k = len(set(topic_model.topics_) - {-1})
            if new_k != current_k - 1:
                print(f"  Warning: Expected k={current_k-1}, got k={new_k}")
            
            # Track merge
            merge_history.append({
                'iteration': api_calls,
                'merged': pair,
                'k_before': current_k,
                'k_after': new_k,
                'llm_time': llm_time,
                'merge_time': merge_time
            })
        
        # Return metrics
        return {
            'merge_history': merge_history,
            'total_api_calls': api_calls,
            'total_llm_time': total_llm_time,
            'avg_llm_time': total_llm_time / api_calls if api_calls > 0 else 0
        }