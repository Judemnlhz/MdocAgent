
from tqdm import tqdm
import importlib
import json
import torch
import os
from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
from agents.verification import (
    apply_verification_decision,
    build_verification_prompt,
    is_refusal_answer,
    parse_verification_response,
    route_question_type,
)

class MDocAgent(MultiAgentSystem):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, question, texts, images):
        question_type = route_question_type(question)
        general_agent = self.agents[-1]
        general_response, messages = general_agent.predict(question, texts, images, with_sys_prompt=True)
        # print("### General Agent: "+ general_response)
        critical_info = general_agent.self_reflect(prompt = general_agent.config.agent.critical_prompt, add_to_message=False)
        # print("### General Critical Agent: " + critical_info)

        start_index = critical_info.find('{') 
        end_index = critical_info.find('}') + 1 
        critical_info = critical_info[start_index:end_index]
        text_reflection = ""
        image_reflection = ""
        try:
            critical_info = json.loads(critical_info)
            text_reflection = critical_info.get("text", "")
            image_reflection = critical_info.get("image", "")
        except Exception as e:
            print(e)

        text_agent = self.agents[1]
        image_agent = self.agents[0]
        all_messages = "General Agent:\n" + general_response + "\n"
        
        relect_prompt = "\nYou may use the given clue:\n"
        text_response, messages = text_agent.predict(question + relect_prompt +text_reflection, texts = texts, images = None, with_sys_prompt=True)
        all_messages += "Text Agent:\n" + text_response + "\n"
        image_response, messages = image_agent.predict(question + relect_prompt +image_reflection, texts = None, images = images, with_sys_prompt=True)
        all_messages += "Image Agent:\n" + image_response + "\n"
            
        # print("### Text Agent: " + text_response)
        # print("### Image Agent: " + image_response)
        final_ans, final_messages = self.sum(all_messages)
        # print("### Final Answer: "+final_ans)
        final_ans, verification_record = self.verify_answer(
            question=question,
            question_type=question_type,
            candidate_answer=final_ans,
            all_messages=all_messages,
        )
        final_messages = {
            "summarizer": final_messages,
            "verification": verification_record,
        }
        
        return final_ans, final_messages

    def verify_answer(self, question, question_type, candidate_answer, all_messages):
        verification_config = self._get_verification_config()
        if not verification_config.get("enabled", False):
            return candidate_answer, {"enabled": False}

        refuse_answer = verification_config.get("refuse_answer", "Not answerable")
        if is_refusal_answer(candidate_answer):
            return refuse_answer, {
                "enabled": True,
                "question_type": question_type,
                "action": "REFUSE",
                "reason": "Candidate answer is already a refusal or empty.",
            }

        verification_prompt = build_verification_prompt(
            question=question,
            question_type=question_type,
            candidate_answer=candidate_answer,
            agent_evidence=all_messages,
            refuse_answer=refuse_answer,
        )
        verifier_response, verifier_messages = self._predict_without_history(
            self.sum_agent,
            verification_prompt,
        )
        decision = parse_verification_response(verifier_response)
        verified_answer = apply_verification_decision(
            candidate_answer=candidate_answer,
            decision=decision,
            refuse_answer=refuse_answer,
        )
        return verified_answer, {
            "enabled": True,
            "question_type": question_type,
            "candidate_answer": candidate_answer,
            "verified_answer": verified_answer,
            "decision": decision,
            "messages": verifier_messages,
        }

    def _predict_without_history(self, agent: Agent, question):
        previous_messages = agent.messages
        agent.messages = None
        try:
            response, messages = agent.predict(question, with_sys_prompt=False)
        finally:
            agent.messages = previous_messages
        return response, messages

    def _get_verification_config(self):
        if hasattr(self.config, "get"):
            verification_config = self.config.get("verification", {})
        else:
            verification_config = getattr(self.config, "verification", {})
        if verification_config is None:
            return {}
        if hasattr(verification_config, "items"):
            return dict(verification_config.items())
        return verification_config
