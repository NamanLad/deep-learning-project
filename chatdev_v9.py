import json
import tiktoken
from typing import Dict, List, Optional
from dataclasses import dataclass
from groq import Groq

tokenizer = tiktoken.get_encoding("cl100k_base")

def truncate_text(text: str, max_tokens: int = 2000) -> str:
    """
    Truncate text to a specified number of tokens
    """
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens) + "\n\n[TRUNCATED]"
    return text

@dataclass
class Message:
    role: str 
    content: str

class AgentMemory:
    """Stores conversation history for each agent with weighted memory"""
    def __init__(self, max_memory_size: int = 3):
        self.messages: List[Message] = []
        self.weights: List[float] = []
        self.max_memory_size = max_memory_size
    
    def add_message(self, role: str, content: str, weight: float = 1.0):
        # Truncate content to manage token size
        content = truncate_text(content, max_tokens=500)
        
        # If memory is full, remove the oldest message
        if len(self.messages) >= self.max_memory_size:
            self.messages.pop(0)
            self.weights.pop(0)
        
        self.messages.append(Message(role, content))
        self.weights.append(weight)
    
    def get_weighted_memory(self) -> str:
        """
        Retrieve memory with more recent messages having higher importance
        """
        if not self.messages:
            return ""
        
        # Limit total memory size
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.messages[-3:])

class Agent:
    """Base agent class with memory"""
    def __init__(self, name: str, role: str, client: Groq, model: str = "llama3-8b-8192"):
        self.name = name
        self.role = role
        self.model = model
        self.client = client
        self.memory = AgentMemory()
    
    def respond(self, prompt: str, memory_weight: float = 1.0) -> str:
        """Generate a response using the LLM, incorporating memory"""
        # Truncate prompt
        prompt = truncate_text(prompt, max_tokens=2000)
        
        self.memory.add_message("user", prompt, memory_weight)
        
        system_prompt = (
            f"You are {self.name}, the {self.role}. {self._get_role_instructions()}\n\n"
            "Recent Context:\n"
            f"{self.memory.get_weighted_memory()}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,  # Limit response tokens
                temperature=0.5
            )
            
            reply = response.choices[0].message.content
            
            reply = truncate_text(reply, max_tokens=2000)
            
            self.memory.add_message(self.name, reply, memory_weight + 0.5)
            
            return reply
        except Exception as e:
            print(f"Error in response generation: {e}")
            return f"Error: {str(e)}"

class CEO(Agent):
    """Chief Executive Officer agent - defines requirements"""
    def _get_role_instructions(self) -> str:
        return (
            "Define clear, concise software requirements. "
            "Focus on key features and critical specifications."
        )

class Coder(Agent):
    """Programmer agent - writes and improves code"""
    def _get_role_instructions(self) -> str:
        return (
            "Write clean, efficient Python code. "
            "Implement core functionality with minimal complexity. "
            "Prioritize readability and essential features."
        )

class Tester(Agent):
    """Tester agent - finds bugs"""
    def _get_role_instructions(self) -> str:
        return (
            "Perform quick, focused testing. "
            "Identify critical bugs and major issues. "
            "Provide succinct, actionable feedback."
        )

class Reviewer(Agent):
    """Code reviewer - ensures quality"""
    def _get_role_instructions(self) -> str:
        return (
            "Review code efficiently. "
            "Focus on major improvements. "
            "Provide clear, concise suggestions."
        )

class Workspace:
    """Holds the project files and state"""
    def __init__(self):
        self.files: Dict[str, str] = {}
        self.requirements: str = ""
        self.tests: str = ""
    
    def add_file(self, filename: str, content: str):
        # Truncate file content
        content = truncate_text(content, max_tokens=3000)
        self.files[filename] = content
    
    def get_file(self, filename: str) -> Optional[str]:
        return self.files.get(filename)

class MiniChatDev:
    """Main controller for the multi-agent system with iterative improvement"""
    def __init__(self, client: Groq):
        self.client = client
        self.reset_agents()
        self.workspace = Workspace()
    
    def reset_agents(self):
        """Reinitialize agents"""
        self.agents = {
            "ceo": CEO("Alice", "CEO", self.client),
            "coder": Coder("Bob", "Lead Developer", self.client),
            "tester": Tester("Charlie", "QA Engineer", self.client),
            "reviewer": Reviewer("Dana", "Senior Reviewer", self.client)
        }
    
    def run_project(self, user_prompt: str, max_iterations: int = 2) -> Dict[str, str]:
        """Execute the full software development lifecycle with iterative improvements"""

        results = []
        # Truncate user prompt
        user_prompt = truncate_text(user_prompt, max_tokens=500)
        
        self.reset_agents()
        
        # Phase 1: Requirements Gathering
        requirements = self._gather_requirements(user_prompt)
        self.workspace.requirements = requirements
        
        # Phase 2: Initial Coding
        code = self._write_code(requirements)
        self.workspace.add_file("main.py", code)
        
        iteration = 0
        while iteration < max_iterations:
            # Phase 3: Testing
            test_report = self._test_code(code)
            
            # Phase 4: Review
            review_report = self._review_code(code)
            
            if "No significant changes" in review_report:
                break
            
            # Improve code based on review suggestions
            improvement_prompt = (
                f"Previous Code (truncated):\n{code[:1000]}\n\n"
                f"Review Suggestions:\n{review_report}\n\n"
                "Modify code addressing these suggestions."
            )

            results.append({"code": code, "test_report": test_report, "review_report": review_report})
            
            # print(f"\n=== Iteration {iteration + 1} ===")
            # print(f"\nCode:\n{code}")
            # print(f"\nTest Report:\n{test_report}")
            # print(f"\nReview Suggestions:\n{review_report}")
            # Increase memory weight for recent suggestions
            code = self.agents["coder"].respond(improvement_prompt, memory_weight=1.5)

            iteration += 1
        
        return {
            "requirements": requirements,
            "final_code": code,
            "test_report": test_report,
            "review_report": review_report,
            "iterations": iteration
        }, results
    
    def _gather_requirements(self, prompt: str) -> str:
        """CEO defines requirements"""
        ceo_prompt = (
            f"User wants to build: {prompt}\n"
            "Provide concise requirements:\n"
            "- Key features\n"
            "- Core specifications"
        )
        return self.agents["ceo"].respond(ceo_prompt)
    
    def _write_code(self, requirements: str) -> str:
        """Coder implements the requirements"""
        coder_prompt = (
            f"Implement in Python:\n{requirements}\n"
            "Provide clean, efficient code."
        )
        return self.agents["coder"].respond(coder_prompt)
    
    def _test_code(self, code: str) -> str:
        """Tester validates the code"""
        test_prompt = (
            f"Test this code (truncated):\n{code[:1000]}\n"
            "Identify critical bugs. Brief report."
        )
        return self.agents["tester"].respond(test_prompt)
    
    def _review_code(self, code: str) -> str:
        """Reviewer checks code quality"""
        review_prompt = (
            f"Review code (truncated):\n{code[:1000]}\n"
            "Highlight key improvements needed."
        )
        return self.agents["reviewer"].respond(review_prompt)

def main():
    client = Groq(api_key="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")
    
    chat_dev = MiniChatDev(client)
    
    print("ChatDev - AI Software Development Team")
    print("------------------------------------------")
    
    user_request = input("What software would you like to build? ")
    
    print("\nStarting development process...")
    results, intermediate_results = chat_dev.run_project(user_request)
    
    # print("\n=== Development Results ===")
    # print(f"\nRequirements:\n{results['requirements']}")
    # print(f"\nFinal Generated Code:\n{results['final_code']}")
    # print(f"\nTest Report:\n{results['test_report']}")
    # print(f"\nCode Review:\n{results['review_report']}")
    # print(f"\nTotal Iterations: {results['iterations']}")

    print("\n=== Development Results ===")
    print(f"\nRequirements:\n{results['requirements']}")
    
    for i, result in enumerate(intermediate_results, start=1):
        print(f"\n=== Iteration {i} ===")
        print(f"\nCode:\n{result['code']}")
        print(f"\nTest Report:\n{result['test_report']}")
        print(f"\nReview Suggestions:\n{result['review_report']}")
    
    # Save results to files
    with open("output.txt", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to output.txt")

if __name__ == "__main__":
    main()