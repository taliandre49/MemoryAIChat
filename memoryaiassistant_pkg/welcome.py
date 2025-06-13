import streamlit as st
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# testing
# Add Font Awesome library for icons
st.markdown("""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Custom CSS for button styling
st.markdown("""
    <style>
    .button-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 50px;
    }
    
    .big-square-button {
        width: 200px;
        height: 100px;
        # background-color: #8A2BE2;
        color: #8A2BE2;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        margin-top: 50px;
        transition: background-color 0.3s ease;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 10px;
    }

    .icon {
        font-size: 40px;
    }

    .button-text {
        font-size: 16px;
        margin-top: 10px;
    }

    .stButton > button {
        width: 100%;  
    }

    .stColumn {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    </style>
""", unsafe_allow_html=True)
st.title("Déjà vu!")

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def redirect_to_page(page_name):
    st.session_state.page = page_name
    st.rerun() 

if st.session_state.page == 'home':
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown("""
                <div class="big-square-button">
                    <i class="fas fa-pen icon"></i> 
                </div>
            """, unsafe_allow_html=True)
            if st.button('Log a Memory', key='log_memory_button'):
                st.switch_page('pages/memory_log.py')

    with col2:
        with st.container():
            st.markdown("""
                <div class="big-square-button">
                    <i class="fas fa-stopwatch icon"></i>  
                </div>
            """, unsafe_allow_html=True)
            if st.button('Open Your Time Capsule', key='open_time_capsule_button'):
                st.switch_page('pages/time_capsule.py')

elif st.session_state.page == 'log_memory':
    st.title('Log a Memory')
    st.write("This is where you can log a memory.")

elif st.session_state.page == 'time_capsule':
    st.title('Open Time Capsule')
    st.write("This is where you can open the time capsule.")


st.subheader("How It Works")
st.markdown("""
1. **Log a memory** by uploading a picture and having a conversation with our AI about it.
2. Use the **time capsule** feature to learn about a past memory.
""")

st.subheader("Thanks for visiting!")
st.write("Hope You Enjoy!")