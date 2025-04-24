import streamlit as st
import candidate
import recruiter
import ATS

# Sidebar for navigation
st.sidebar.title("Internova.app") 
st.sidebar.subheader("End-to-End Job Search Platforms")  
page = st.sidebar.radio("Go to", ("Home", "Candidate", "Recruiter", "ATS Score"))

# Adding the note at the bottom of the sidebar in one paragraph
st.sidebar.write("Note: This site is currently a prototype. We acknowledge that users may encounter some unexpected errors when navigating to pages. These issues can typically be resolved by refreshing the page")

# Home page content
if page == "Home":
    # Streamlit UI
    st.markdown(
        """
        <style>
        .center-text {
            text-align: center;
        }
        .spacer {
            margin-top: 40px;  /* Adjust this value to increase or decrease the space */
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='center-text'>INTERNOVA</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='center-text'>Where Skills Meet Opportunity</h2>", unsafe_allow_html=True)
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    st.subheader("Our Vision")
    st.write("""
    Internova.app is your go-to platform for all job search and recruitment needs. Whether you're a candidate looking for your next opportunity 
    or a recruiter seeking the perfect match, our end-to-end solutions cover it all. With easy-to-use resume-building tools, smart job and candidate 
    recommendations, and an ATS score checker, we are here to bridge the gap between talent and opportunity.
    
    Navigate using the sidebar to explore different sections.
    """)


# Candidate page
elif page == "Candidate":
    candidate.app()

# Recruiter page
elif page == "Recruiter":
    recruiter.app()

# ATS Score page
elif page == "ATS Score":
    ATS.app()