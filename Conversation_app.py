from langchain_core.prompts import ChatPromptTemplate
import tiktoken
import json
from datetime import datetime
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from retriver import retrieve_answer 
#from langchain_core.chat_history import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Default Configuration
#DEFAULT_API_KEY = os.environ.get("TOGETHER_API_KEY")
#DEFAULT_BASE_URL = "https://api.together.xyz/v1"
#DEFAULT_MODEL = "gpt-3.5-turbo"

class ConversationManager:
    def __init__(self,temprature = 0.7,max_tokens = 256,max_history = 5,history_file = None):

    # Initialize LangChain Chat Mode
        self.temprature = temprature
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI(
            model = 'gpt-3.5-turbo',temperature = self.temprature,max_tokens = self.max_tokens
        )
        self.system_messages = {
            "financial_expert": """You are a highly knowledgeable financial expert helping users with banking queries.
            You provide clear, accurate, and up-to-date information about banking, loans, investments, and fraud prevention.
            You do not provide legal or personalized financial advice. Use the following retrieved information to answer the query:\n\n{context}"""
        }
        self.system_message = self.system_messages['financial_expert']

        self.chat_history = []
        self.max_history = max_history
        if history_file is None :
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.history_file = f"conversation_history_{timestamp}.json"
        else:
            self.history_file = history_file
        self.load_conversation_history()
    
    def enforce_history_limit(self):
        while len(self.chat_history) > self.max_history + 1:
            self.chat_history.pop(1)
     
    def count_tokens(self,text):

        try :
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')

        tokens = encoding.encode(text)
        return len(tokens)
    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.chat_history)
        
    def set_persona(self,persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")
    def set_custom_system_message(self, custom_message):
        """
        Allows the user to define a custom system message.
        If empty, it raises an error.
        """
        if not custom_message:
            raise ValueError("Custom message cannot be empty.")
    
        self.system_messages['custom'] = custom_message  # Store it under 'custom'
        self.set_persona('custom')

    def update_system_message_in_history(self):
        if self.chat_history and isinstance(self.chat_history[0], SystemMessage):
            self.chat_history[0] = SystemMessage(content=self.system_message)  # Update existing system message
        else:
            self.chat_history.insert(0, SystemMessage(content=self.system_message))
    
    def load_conversation_history(self):
        try:
            with open(self.history_file,'r') as file:
                self.chat_history = json.load(file)
                self.chat_history = [SystemMessage(content=msg['content']) if msg['role'] == 'system' 
                                    else HumanMessage(content=msg['content']) if msg['role'] == 'human' 
                                    else AIMessage(content=msg['content']) for msg in self.chat_history]
        except FileNotFoundError:
            self.chat_history  = [SystemMessage(content=self.system_message)]
            
        except json.JSONDecodeError:
            print("Error reading the conversation history file.")
            self.chat_history  = [SystemMessage(content=self.system_message)]

    def save_conversation_history(self):
        with open(self.history_file, "w") as file:
            json.dump([{"role": "system" if isinstance(msg, SystemMessage) 
                        else "human" if isinstance(msg, HumanMessage) 
                        else "ai", 
                        "content": msg.content} for msg in self.chat_history], 
                    file, indent=4)

    def chat_completion(self, user_input):

        #print(len(self.chat_history))
        self.enforce_history_limit()
        retrived_answer = retrieve_answer(user_input)
        if len(retrived_answer) == 0:
           self.chat_history.append(HumanMessage(user_input))
           self.chat_history.append(AIMessage("⚠️ Sorry, I couldn't find any relevant information for your query."))
           self.save_conversation_history()
           return "⚠️ Sorry, I couldn't find any relevant information for your query."
        else:
            context = " ".join([" ".join(chunk["answers"]) for chunk in retrived_answer])  
            #self.chat_history.append(HumanMessage(content=f"User Query: {user_input}\n\nRelevant Information: {context}"))

        # Define Prompt Template with Chat History
            prompt_template = ChatPromptTemplate.from_messages([
                        SystemMessage(content=self.system_message),
                    HumanMessage(content=f"{user_input}")
                                        ])
       # prompt_template = ChatPromptTemplate.from_messages([
          #  SystemMessage(content="You are an expert Advisor of Life."),
           # HumanMessage(content=user_input)  # Using 'prompt' as variable
        #])
            chain = prompt_template | self.llm
            response = chain.invoke({})  # Use .invoke() instead of .run()
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response.content))
            self.enforce_history_limit()
        #print(self.chat_history)
        #print(len(self.chat_history))
            self.save_conversation_history()
            return response.content
    def reset_conversation_history(self):
        self.chat_history = [SystemMessage(content=self.system_message)]
        






