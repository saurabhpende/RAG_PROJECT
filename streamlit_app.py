import streamlit as st
from app import ConversationManager
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Initialize Chatbot
#conv_manager = ConversationManager()
#conv_manager.set_persona('financial_expert')

# Streamlit UI
#st.set_page_config(page_title="💰 Financial AI Chatbot", layout="wide")
st.title("Financial Expert AI Chatbot 💰")
#st.subheader("Ask me anything about banking, investments, and loans!")
st.sidebar.header("Options")
if 'chat_manager' not in st.session_state:
    st.session_state['chat_manager'] = ConversationManager()
chat_manager = st.session_state['chat_manager']

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
chat_manager.set_persona('financial_expert')
#if st.sidebar.button("Reset conversation history", on_click=chat_manager.reset_conversation_history()):
   # st.session_state['chat_history'] = chat_manager.chat_history

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = chat_manager.chat_history

chat_history = st.session_state['chat_history']

user_input = st.chat_input("Write a message")

if user_input:
    response = chat_manager.chat_completion(user_input)




# Display Chat History from LangChain
#st.write("### 📝 Conversation History")
for message in chat_manager.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)  # Displays user query
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)  # Displays AI response


# User Input
#user_input = st.text_input("💬 Type your financial question here...", key="query")

#if st.button("🔎 Get Answer"):
   # if not user_input.strip():
   #     st.warning("⚠️ Please enter a valid question.")
   # else:
   #     st.write("⏳ Retrieving relevant financial data...")

    #    # Get AI Response
     #   response_text = conv_manager.chat_completion(user_input)

     #   # Display Response
      #  st.success("✅ Answer Generated!")
      #  st.markdown(f"**🤖 AI:** {response_text}")

# Clear Chat Button
#if st.sidebar.button("🗑️ Clear Chat"):
   # conv_manager.chat_history = [SystemMessage(content=conv_manager.system_message)]
   # st.experimental_rerun()
