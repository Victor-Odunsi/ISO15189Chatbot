import streamlit as st 
import uuid
from datetime import datetime
import json
from client import get_api_response_stream

st.set_page_config(
    page_title = 'QMS Assistant',
    layout = 'centered'
    )

st.markdown(
    """
    <style>
    /* Center app content */
    .block-container {
        text-align: center;
    }
    """,
    unsafe_allow_html=True
)



st.header('🤖 Hi, Where should we begin?')


@st.cache_resource
def get_persistent_store():
    return {}

# Initialize session state
def initialize_session_state():

    store = get_persistent_store()

    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())

def save_current_session():
    """Save current messages to the sessions dictionary"""
    if st.session_state.messages:
        # Create a title from the first user message (truncated)
        first_message = next((msg['content'] for msg in st.session_state.messages if msg['role'] == 'user'), "New Chat")
        title = first_message[:50] + "..." if len(first_message) > 50 else first_message
        
        st.session_state.chat_sessions[st.session_state.current_session_id] = {
            'title': title,
            'messages': st.session_state.messages.copy(),
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.session_id
        }

def load_session(session_key):
    """Load a specific session"""
    if session_key in st.session_state.chat_sessions:
        session_data = st.session_state.chat_sessions[session_key]
        st.session_state.messages = session_data['messages'].copy()
        st.session_state.current_session_id = session_key
        st.session_state.session_id = session_data.get('session_id')
        st.rerun()

def start_new_chat():
    """Start a new chat session"""
    save_current_session()  # Save current session before starting new one
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.session_id = None
    st.session_state.messages = []
    st.rerun()

def delete_session(session_key):
    """Delete a specific session"""
    if session_key in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[session_key]
        
        # If we deleted the current session, start a new one
        if session_key == st.session_state.current_session_id:
            start_new_chat()
        else:
            st.rerun()

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        now = datetime.now()
        
        # If today, show time
        if dt.date() == now.date():
            return dt.strftime("%H:%M")
        # If this week, show day
        elif (now - dt).days < 7:
            return dt.strftime("%a")
        # Otherwise show date
        else:
            return dt.strftime("%m/%d")
    except:
        return ""

# Initialize the app
initialize_session_state()

# Sidebar
with st.sidebar:
    st.title("💬 Chat History")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
    
    st.divider()
    
    # Chat sessions list
    if st.session_state.chat_sessions:
        st.subheader("Recent Chats")
        
        # Sort sessions by timestamp (most recent first)
        sorted_sessions = sorted(
            st.session_state.chat_sessions.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        for session_key, session_data in sorted_sessions:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Create a button for each session
                is_current = session_key == st.session_state.current_session_id
                button_type = "primary" if is_current else "secondary"
                
                if st.button(
                    f"{'🟢 ' if is_current else ''}{session_data['title']}", 
                    key=f"load_{session_key}",
                    help=f"Created: {format_timestamp(session_data['timestamp'])}",
                    use_container_width=True,
                    type=button_type
                ):
                    if not is_current:
                        load_session(session_key)
            
            with col2:
                # Delete button
                if st.button("🗑️", key=f"del_{session_key}", help="Delete chat"):
                    delete_session(session_key)
    
    else:
        st.info("No chat history yet. Start a conversation!")

# Auto-save current session when messages change
if st.session_state.messages:
    save_current_session()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask me anything")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.status('Just a sec...', expanded=False) as status_box:
            for response in get_api_response_stream(prompt, st.session_state.session_id):
                if response['type'] == 'session':
                    st.session_state.session_id = response['session_id']
                    status_box.update(label="Processing your request...", state="running")
                    
                elif response['type'] == 'token':
                    full_response += response['content']
                    status_box.update(label="Generating response...", state="running")
                    message_placeholder.markdown(full_response + "▌")
                    
                elif response['type'] == 'end':
                    status_box.update(label="Done!", state="complete")
                    break
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        message_placeholder.markdown(full_response)

# Display session info in footer (optional)
with st.container():
    session_display = st.session_state.session_id[:8] + "..." if st.session_state.session_id else "None (new chat)"
    st.caption(f"Session ID: {session_display} | Messages: {len(st.session_state.messages)}")