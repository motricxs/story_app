import streamlit as st
from transformers import pipeline
import torch

# --- Configuration and Model Loading ---

# Set the page configuration. This is the first command that must be run.
st.set_page_config(
    page_title="Story Maker",
    page_icon="📖",  # A nice icon for the page
    layout="wide"    # Use the full width of the page
)

# Use st.cache_resource to load the model only once.
# This makes the app run much faster after the first load.
@st.cache_resource
def load_story_generator():
    """
    Loads the story generation pipeline from Hugging Face.
    """
    # Using a pipeline simplifies the code for text generation.
    # It will use the GPU if available (device=0), otherwise the CPU (device=-1).
    story_pipe = pipeline("text-generation",
                          model="roneneldan/TinyStories-Instruct-33M",
                          device=0 if torch.cuda.is_available() else -1)
    return story_pipe

# Load the model and show a status message to the user.
try:
    story_generator = load_story_generator()
except Exception as e:
    # If the model fails to load, show an error and stop the app.
    st.error(f"Error: Could not load the story engine. Please refresh. Details: {e}")
    st.stop()


# --- User Interface (UI) ---

# A clean, professional title with a divider.
st.title("📖 AI Story Maker")
st.divider()

# Organize the layout with sidebar for settings and main area for content.
# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Control the story's details here.")

    # Slider for story length.
    max_length = st.slider(
        "Story Length",
        min_value=50,
        max_value=300,
        value=150,
        step=10,
        help="How long should the story be (in words)?"
    )

    # Slider for creativity (temperature).
    creativity = st.slider(
        "Creativity (Temperature)",
        min_value=0.5,
        max_value=1.5,
        value=0.9,
        step=0.1,
        help="Higher values make the story more creative, but maybe less logical."
    )
    
    st.divider()
    st.info("Made with ❤️ by PoulStar")

    # --- LOGO SECTION: UPDATED FOR SIZING AND CENTERING ---
    logo_url = "https://raw.githubusercontent.com/poulstar/.github/main/logo.png"

    # We create 3 columns in the sidebar to center the logo.
    # The image is placed in the middle column (col2).
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # The width is set to 250px to be large but still fit inside the sidebar.
        st.image(logo_url, width=250)


# --- Main Content Area ---

# Use columns for a cleaner layout
col1, col2 = st.columns([0.6, 0.4]) # Give more space to the input area

with col1:
    st.subheader("What is your story about?")
    
    # Text area for the user's prompt with a placeholder.
    prompt = st.text_area(
        label="Write your story idea here. The AI will write a story for you.",
        placeholder="For example: A friendly robot that likes to garden.",
        height=150,
        label_visibility="collapsed" # Hides the label to save space
    )

    # Button to trigger story generation.
    generate_button = st.button("✨ Write My Story!", type="primary", use_container_width=True)


# --- Logic for Story Generation ---

# We will only show the story output in the second column if a story has been generated.
with col2:
    if generate_button:
        if prompt:
            # Show a spinner while the model is working.
            with st.spinner("An AI author is writing your story..."):
                try:
                    # Generate the story using the pipeline.
                    generated_story = story_generator(
                        prompt,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=creativity,
                        do_sample=True
                    )[0]['generated_text']
                    
                    # Store the generated story in the session state
                    st.session_state['story'] = generated_story

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            # Show a warning if the user clicks the button without entering a prompt.
            st.warning("Please enter your story idea first!")

    # Display the story if it exists in the session state
    if 'story' in st.session_state and st.session_state['story']:
        st.subheader("Your New Story:")
        # Use a container with a border for a nice visual effect.
        with st.container(border=True):
            st.markdown(st.session_state['story'])