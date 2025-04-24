import streamlit as st
import spacy
from spacy.matcher import Matcher
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import base64
def app():
    # Initialize resources once
    if 'initialized' not in st.session_state:
        model_path = os.path.join(os.path.dirname(__file__), 'models/en_core_web_sm-3.7.1')
        st.session_state['nlp'] = spacy.load(model_path)
    
        # List of skills
        skills = ['Gesture Recognition', 'Penetration Testing Tools', 'Risk Management', 'Virtual Private Network', 'Cyber Threat Intelligence', 'Sales Strategy', 'Edge Computing Frameworks',
               '5G Networks', 'Clustering', 'Productivity Improvement', 'Asynchronous Programming', 'Stakeholder Management', 'Marketing Strategy', 'Data Quality Management', 'Employee Engagement',
               'Scikit-Learn', 'SMTP', 'Marketing', 'Communication', 'Edge Computing Security', 'Time-Series Forecasting', 'Coaching', 'CD', 'Mobile Development', 'Unsupervised Learning', 'Leadership',
               'Team Motivation', 'MFA', 'Business Process Automation', 'RPC', 'CDN', 'Robustness Testing', 'Embedded Systems', 'NAS', 'Leadership Strategy', 'Empathy', 'AutoML', 'Agile', 'Decision Making Frameworks', 'Blockchain',
               'Behavioral Interviewing', 'Integration Testing', 'AWS', 'Digital Twins Technology', 'Microservices', 'Subversion', 'Strategic Partnerships', 'Big Data',
               'Data Ethics', 'Analytical Skills', 'Time Series Analysis', 'ETL', 'Process Documentation', 'OAuth2', 'Workplace Etiquette', 'Disaster Recovery Planning',
               'Power BI', 'Data Lineage', 'Decision Making', 'Process Improvement', 'Site Reliability Engineering', 'Cloud-Native Security', 'Automated Testing', 'Performance Management',
               'DDoS', 'SIEM', 'Data Lakes', 'Unix', 'XAI', 'Reinforcement Learning', 'Data Analysis', 'Crisis Management', 'Security', 'Terraform', 'PyTorch', 'Face Recognition', 'Prometheus', 'Decision Analysis', 'Cloud Migration', 'GPA', 'Exploratory Data Analysis', 'Jupyter Notebooks', 'High Performance Computing',
               'Ansible', 'CloudOps', 'General Data Protection Regulation', 'VCS', 'Continuous Integration/Continuous Deployment', 'Explainable AI', 'Resource Management',
               'Function as a Service', 'Biometric Security', 'Emotional Intelligence', 'Deep Learning', 'Hadoop', 'Facilitation Skills', 'ORM', 'Strategic Planning', 'Program Development',
               '3D Printing', 'BI', 'Scripting', 'Google Analytics', 'Presentation Skills', 'Model-View-Controller', 'Business Communication', 'CICD', 'Google Cloud Platform', 'Knowledge Graphs',
               'HIPAA', 'Blockchain for Supply Chain', 'Data-Driven Decision Making', 'Negotiation Skills', 'Networking', 'Organizational Change', 'Machine Translation', 'Natural Language Generation',
               'Resource Planning', 'ELK Stack', 'Cross-Functional Teams', 'Hybrid Cloud', 'Trend Analysis', 'Crisis Intervention', 'Assertiveness', 'Algorithmic Trading', 'Software Architecture', 'Cloud Native Development',
               'OAuth', 'Micro Frontends', 'MLOps', 'DMS', 'UDP', 'Tableau', 'CSS', 'Voice User Interface', 'Employee Relations', 'Event Planning', 'Multi-factor Authentication', 'Business Acumen', 'Team Collaboration',
               'Customer Service', 'Database Management System', 'Strategic Negotiation', 'Feedback Management', 'Computer-Aided Design', 'JSON Web Token', 'Cloud Cost Management', 'Support Vector Machines', 'Work-Life Balance',
               'Identity and Access Management', 'Networking Strategies', 'Serverless Computing Security', 'Supply Chain Management', 'Software Development Life Cycle', 'SQL', 'Relational Database Management System', 'Nagios',
               'Data Normalization', 'Puppet', 'Internet Protocol', 'PySpark', 'Hypothesis Testing', 'Pandas', 'CI/CD Pipelines', 'Raspberry Pi', 'Java', 'Public Speaking Engagements', 'Supervised Learning', 'JIRA', 'GPU Programming',
               'Public Speaking', 'Team Building', 'Data Imputation', 'Market Research', 'Dependency Injection', 'Data Cleaning', 'Project Coordination', 'Command Line Interface', 'Influencing', 'Incident Response', 'Version Control System',
               'Change Management', 'Content Delivery Network', 'Real-Time Data Processing', 'CV', 'Apache Spark', 'Augmented Reality', 'Amazon Web Services', 'Organizational Effectiveness', 'Random Forest', 'VLAN', 'Strategic Thinking',
               'Code Quality', 'Team Effectiveness', 'Privacy-Preserving Machine Learning', 'Software-defined Networking', 'SaaS', 'Data Security', 'HPC', 'Containerization', 'Test-Driven Development', 'RDBMS', 'Secure Coding Practices', 'API', 'Organizational Development', 'Container Security',
               'Speech Recognition', 'NumPy', 'Distributed Systems Design', 'Organizational Agility', 'Workflow Management', 'Collaboration Tools', 'EAI', 'File Transfer Protocol', 'Goal Achievement', 'Time Prioritization', 'Data Wrangling',
               'Community Engagement', 'ANN', 'XGBoost', 'Cultural Awareness', 'PD', 'Writing Skills', 'Training Development', 'Operational Strategy', 'Cassandra', 'A/B Testing', 'Load Balancing', 'Matplotlib', 'Backend as a Service', 'XML',
               'Secure Multi-Party Computation', 'Teamwork', 'GCP', 'Leadership Development', 'Organizational Behavior', 'Infrastructure as Code', 'HTTPS', 'Kubernetes', 'Storage Area Network', 'Linux', 'Performance Metrics', 'Stakeholder Engagement',
               'Hybrid Cloud Strategies', 'Quantum Cryptography', 'Digital Analytics', 'Active Listening', 'GraphQL', 'Static Code Analysis', 'Business Continuity Planning', 'GANs', 'Market Positioning', 'Conversational AI', 'Data Federation', 'JS',
               'Problem Solving', 'Negotiation', 'Goal Setting', 'C++', 'Go', 'Change Facilitation', 'Regression', 'Service Mesh', 'VR', 'Data Governance Frameworks', 'SDLC', 'Internet Message Access Protocol', 'Innovation Facilitation', 'SSH', 'Persuasion', 'NoSQL', '5G Security',
               'Decision Trees', 'Client Engagement', 'Knowledge Management Systems', 'Biostatistics', 'Organizational Strategy', 'Employee Onboarding', 'Application Programming Interface', 'Data Version Control', 'Medical Imaging Analysis', 'Columnar Databases', 'Research', 'Ethical Judgment',
               'Unit Testing', 'Application Security Testing', 'Human-Computer Interaction', 'DBMS', 'Selenium', 'Regression Analysis', 'Sponsorship Management', 'Data Visualization',
               'Data Integration', 'API Gateway', 'Network Monitoring', 'Emotional Resilience', 'Service-Level Agreements', 'Kotlin', 'Immutable Infrastructure', 'Document-Based Databases', 'Data Center Infrastructure', 'Customer Data Platforms', 'Data Augmentation', 'Long Short Term Memory',
               'Advanced Persistent Threats', 'R', 'IMAP', 'Hyperparameter Tuning', 'Computer Vision', 'Collaboration', 'Software as a Service', 'Graphical User Interface', 'Basically Available, Soft state, Eventually consistent', 'Excel', 'Organizational Skills', 'Workforce Planning', 'CAP',
               'Network Function Virtualization', 'Mixed Reality', 'VoIP', 'Product Management', 'Penetration Testing', 'Network Address Translation', 'Sales', 'Hypertext Transfer Protocol', 'Cloud Orchestration', 'Serverless', 'Internet of Things', 'Zero-Knowledge Proofs', 'Threat Modeling',
               'Remote Procedure Calls', 'Self-Healing Systems', 'RESTful APIs', 'Data Loss Prevention', 'Computational Fluid Dynamics', 'Numerical Methods', 'Git', 'Feature Engineering', 'Behavioral Analytics', 'Relationship Building', 'Azure', 'Log Management', 'Training', 'Network Attached Storage',
               'Conflict Management', 'Metadata Management', 'Robotic Process Automation', 'VUI', 'Network Security', 'MATLAB', 'Grafana', 'Automation', 'Customer Experience Management', 'Principal Component Analysis', 'Networking Skills', 'Integrated Development Environment', 'BaaS', 'Public Cloud',
               'Change Readiness', 'Virtual Local Area Network', 'Dashboard Creation', 'Mentorship', 'Change Leadership', 'Domain-Driven Design', 'Media Relations', 'Diversity and Inclusion', 'Incident Management', 'Process Optimization', 'Public Speaking Preparation', 'Resilience Engineering', 'CAD',
               'Virtual Reality', 'Federated Learning', 'Load Testing', 'PaaS', 'Chef', 'SLO', 'Conflict Resolution', 'Interpersonal Skills', 'Delegation', 'CLI', 'Self-Motivation', 'Customer Insights', 'Editing', 'Keras', 'Hypertext Transfer Protocol Secure', 'SVN', 'Salesforce', 'OS', 'FaaS', 'SA',
               'Cybersecurity Frameworks', 'Classification', 'Shell Scripting', 'Time Management', 'SVM', 'Behavior-Driven Development', 'Event Management', 'Data Sharding', 'Edge AI', 'Data Streaming', 'Public Relations', 'DLP', 'PCA', 'Event Coordination', 'Data Privacy Laws', 'AR', 'AI Operations',
               'Genomics Data Analysis', 'SAN', 'Data Privacy', 'Disaster Recovery Automation', 'Data Masking', 'Version Control', 'Blockchain Development', 'Model Evaluation', 'Client Retention Strategies', 'Atomicity, Consistency, Isolation, Durability', 'Create, Read, Update, Delete', 'Mentoring',
               'Apache Kafka', 'GUI', 'User Datagram Protocol', 'Monitoring', 'Data Mining', 'Cross-Validation', 'Digital Fabrication', 'Facilitation', 'Team Leadership', 'JWT', 'Service Excellence', 'NLP', 'Critical Thinking', 'Ethical Hacking', 'MVP', 'CDP', 'Automated Machine Learning',
               'Operational Leadership', 'AI Model Interpretability', 'Sales Enablement', 'DevOps Pipelines', 'Ensemble Learning', 'JavaScript', 'Open Authorization', 'Team Dynamics', 'Graph Databases', 'Brand Management', 'Performance Metrics Analysis', 'Data Pipeline', 'IaC', 'Knowledge Sharing',
               'Innovation Management', 'Generative Adversarial Networks', 'IDE', 'Data Orchestration', 'Natural Language Processing', 'Model Interpretability', 'NoSQL Databases', 'Talent Acquisition', 'Real-Time Analytics', 'Database Management', 'Health Insurance Portability and Accountability Act',
               'Event-Driven Architecture', 'Digital Communication', 'Quantum Computing', 'Wearable Technology', 'Statistical Modeling', 'Database Scaling', 'Virtual Machine', 'GDPR', 'ACID', 'Deep Reinforcement Learning', 'Customer Retention', 'Neural Networks', 'User Experience/User Interface', 'CI/CD',
               'Cross-Platform Development', 'Client Relations', 'Client Feedback', 'Project Management', 'Parallel Computing', 'Android Development', 'Model-Driven Engineering', 'Python', 'Object Detection', 'Leadership Coaching', 'ElasticSearch', 'Brand Positioning', 'Digital Signal Processing',
               'Machine Learning', 'WebSockets', 'Budgeting', 'Cultural Sensitivity', 'Data Visualization Tools', 'Gradient Boosting', 'Digital Transformation Strategy', 'Agile Methodology', 'Low-latency Systems', 'Strategic Visioning', 'Event Sourcing', 'GPU', 'Resource Allocation', 'Infrastructure as a Service',
               'Server-Side Rendering', 'DevSecOps', 'VM', 'Microsoft Excel', 'DataOps', 'Crisis Communication', 'Software-defined Storage', 'ML', 'Server Management', 'Minimum Viable Product', 'Service-Oriented Architecture', 'Cloud Service Providers', 'Algorithm Optimization', 'Dimensionality Reduction',
               'Cryptographic Protocols', 'I/O', 'DeFi', 'Continuous Delivery', 'AIOps', 'Ethical AI', 'Leadership Skills Development', 'Homomorphic Encryption', 'Customer Journey Mapping', 'Google Sheets', 'UX/UI', 'Distributed Denial of Service', 'Sales Management', 'Client Acquisition', 'Scrum', 'Mentoring and Coaching',
               'Workplace Diversity', 'HTML', 'Graph Theory', 'BDD', 'Seaborn', 'Spark', 'KPI', 'Business Intelligence', 'DataViz', 'IP', 'Consistency, Availability, Partition tolerance', 'Customer Relations', 'Performance Evaluation', 'Statistical Analysis', 'Synthetic Data Generation', 'Customer Needs Analysis', 'RPA',
               'WebAssembly', 'Artificial General Intelligence', 'Regulatory Compliance', 'APIs Management', 'Customer Acquisition', 'Hyperparameter Optimization', 'Interactive Dashboards', 'Employee Development', 'Zero Trust Architecture', 'SLA', 'VPN', 'MVC', 'Continuous Monitoring', 'Ethical Hacking Tools', 'Edge Computing',
               'Secure Shell', 'FTP', 'Cloud Cost Optimization', 'Operating System', 'Organizational Leadership', 'NFS', 'Communication Strategy', 'Computational Geometry', 'HBase', 'JSON', 'TDD', 'Voice over Internet Protocol', 'Digital Rights Management', 'LSTM', 'DL', 'TCP', 'Transmission Control Protocol', 'Docker', 'NAT',
               'Dynamic Code Analysis', 'Network File System', 'Creativity', 'HTTP', 'CAP Theorem', 'Business Development', 'Rust', 'TensorFlow', 'Jenkins', 'Intercultural Communication', 'Distributed Ledger Technology', 'IoT', 'Private Cloud', 'ACID Transactions', 'AI', 'Client Management', 'Predictive Analytics', 'IaaS',
               'Input/Output', 'Cybersecurity', 'Component-Based Architecture', 'Predictive Modeling', 'Drone Programming', 'Cloud Security', 'Artificial Intelligence', 'Data Governance', 'Digital Twins', 'Adaptability', 'Multi-Cloud Management', 'Load Balancer', 'Conflict Negotiation', 'Brand Development', 'Presentation Design',
               'Stress Management', 'Artificial Neural Networks', 'Edge Analytics', 'Descriptive Statistics', 'Multithreading', 'Cloud Infrastructure Automation', 'Data Anonymization', 'Decentralized Finance', 'K8s', 'Content Creation', 'Platform as a Service', 'iOS Development', 'Simple Mail Transfer Protocol', 'SciPy', 'Data Warehousing',
               'Big Data Frameworks', 'Neural Architecture Search', 'Orchestration', 'CRUD', 'Operational Efficiency', 'Attention to Detail', 'Object-Relational Mapping', 'BASE', 'Redis']

        # Matcher for skill extraction
        matcher = Matcher(st.session_state['nlp'].vocab)
        for skill in skills:
            skill_patterns = [{'LOWER': term.lower()} for term in skill.split()]
            matcher.add(skill, [skill_patterns])
        st.session_state['matcher'] = matcher

        # Load dataset and Word2Vec model
        file_name = 'Candidate_csv_final_base64.csv'
        file_path = os.path.join(os.getcwd(), file_name)
        df = pd.read_csv(file_path)
        job_skills = df['skillsofusers']
        skills_processed = [item.lower().replace(',', ' ').replace(' ', '_').split() for item in skills]
        word2vec_model = Word2Vec(skills_processed, vector_size=100, window=5, min_count=1)

        st.session_state.update({
        'job_skills': job_skills,
        'df': df,
        'word2vec_model': word2vec_model,
        'initialized': True
        })

    # Skill extraction using matcher
    def skills_extract(text):
        doc = st.session_state['nlp'](text)
        matches = st.session_state['matcher'](doc)
        return list({doc[start:end].text for match_id, start, end in matches})

    # Compute vector for skills list
    def compute_vector(words_list):
        model = st.session_state['word2vec_model'].wv
        vectors = [model[word] for word in words_list if word in model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    # Calculate similarity between user skills and job skills
    def calculate_similarity(user_skills):
        user_vector = compute_vector([skill.lower().replace(' ', '_') for skill in user_skills]).reshape(1, -1)
        return [
            cosine_similarity(user_vector, compute_vector([skill.lower().replace(' ', '_') for skill in job_skill]).reshape(1, -1))[0][0]
            for job_skill in st.session_state['job_skills']
     ]   
    
    # Generate job recommendations based on similarity threshold
    def generate_recommendations(user_data, threshold=0):
        df = st.session_state['df']

    
        sim_scores = calculate_similarity(user_data)
        df_copy = df.copy()
        df_copy['similarity'] = sim_scores
        recommendations = df_copy[df_copy['similarity'] >= threshold].sort_values('similarity', ascending=False)

        return recommendations


    # Streamlit UI for Recruiter
    st.title("INTERNOVA - Recruiter Portal")
    st.write(f'### Find the Right Talent for Your Organization')

    #Recruiter Input Fields
    recruiter_name = st.text_input("Recruiter's Name")
    organization_name = st.text_input("Organization Name")
    candidate_count = st.slider("Number of Candidate Suggestions", min_value=1, max_value=10, value=5)

    # Job description text area
    job_description = st.text_area("Job Description", height=150)
    extracted_skills = skills_extract(job_description)
    st.text_area("Extracted Skills", ', '.join(extracted_skills), height=200)

    # Submit button to get candidate recommendations
    if st.button("Submit", key="submit_button_1"):
        if recruiter_name and organization_name and job_description and extracted_skills:
            # Assuming get_recommendation is the function that takes job description and returns a DataFrame
            recommended_candidates = generate_recommendations(extracted_skills)
        
            # Display candidate recommendations
            st.write(f"### Top {candidate_count} Candidate Recommendations")
        
            for idx, row in recommended_candidates.head(candidate_count).iterrows():
                with st.expander(f"**{row['Name']} - {row['Education']}**", expanded=False):
                    st.write(f"*Date of Birth:* {row['DOB']}")
                    st.write(f"*Email:* {row['Email']}")
                    st.write(f"*Phone Number:* {row['phone_number']}")
                    st.write(f"*Country:* {row['Country']}")
                    st.write(f"*Location:* {row['location']}")
                    st.write(f"*Address:* {row['Address']}")
                    st.write(f"*Gender:* {row['Gender']}")
                    st.write(f"*Zip Code:* {row['Zip_Code']}")
                    # Display the resume as a base64 PDF
                    resume_base64 = row['base64_pdf']
                    pdf_data = base64.b64decode(resume_base64)
                    st.download_button(label="Download PDF", data=pdf_data, file_name=f"{row['Name']}_resume.pdf")
                    st.markdown(f'<iframe src="data:application/pdf;base64,{resume_base64}" width="700" height="400"></iframe>', unsafe_allow_html=True)
        else:    
            st.write("Please fill all the required fields before submitting.")

if __name__ == "__main__":
    app()