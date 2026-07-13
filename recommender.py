# =====================================================================
# PROJECT 3: DIGITAL MATCHMAKER (TECH STACK RECOMMENDER)
# Content-Based Filtering using TF-IDF & Cosine Similarity
# =====================================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- LOAD DATASET ---
# Loading the item features (job roles and their required skills)
try:
    df = pd.read_csv('raw_skills.csv')
except FileNotFoundError:
    print("Error: raw_skills.csv not found. Please ensure it is in the same folder.")
    exit()

print("🤖 Welcome to the AI Tech Stack Recommender!")
print("Let's map your skills to the perfect job role.\n")

# ==========================================
# STEP 1: INGESTION (Bypassing Cold Start)
# ==========================================
# Capturing explicit user state. We require at least 3 skills.
user_skills = []
print("Enter at least 3 skills you possess (e.g., Python, Cloud, Automation):")
for i in range(1, 4):
    skill = input(f"Skill {i}: ").strip()
    user_skills.append(skill)

# Combining user skills into a single string for vectorization
user_profile_text = " ".join(user_skills)
print(f"\n[Ingestion Complete] User Profile: {user_profile_text}")

# ==========================================
# STEP 2: SCORING (TF-IDF & Cosine Similarity)
# ==========================================
# 1. Bridging the language barrier through Vector Mapping
vectorizer = TfidfVectorizer()

# We add the user's profile to the list of job skills to ensure they share the SAME vocabulary space
all_texts = df['Skills'].tolist() + [user_profile_text]

# Generate TF-IDF Matrix (converting words to weighted numbers)
tfidf_matrix = vectorizer.fit_transform(all_texts)

# The user vector is the last item in the matrix, item vectors are everything before it
item_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1]

# 2. Run the similarity math
# Calculating the angular alignment (cosine similarity) between user and all items
similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()

# ==========================================
# STEP 3: SORTING
# ==========================================
# Add the calculated scores back to our dataframe
df['Similarity_Score'] = similarity_scores

# Organize the scored dataset in descending order
df_sorted = df.sort_values(by='Similarity_Score', ascending=False)

# ==========================================
# STEP 4: FILTERING (Preventing Choice Overload)
# ==========================================
# Truncate the output to generate the Top-N list (Top 3 matches)
top_n = 3
top_matches = df_sorted.head(top_n)

# --- OUTPUT THE MATCHES ---
print(f"\n🎯 TOP {top_n} RECOMMENDED CAREER PATHS:")
print("-" * 50)

for index, row in top_matches.iterrows():
    role = row['Job_Role']
    score = row['Similarity_Score'] * 100 # Converting to percentage
    matched_skills = row['Skills']
    
    # Only show relevant roles with a score greater than 0 (orthogonal)
    if score > 0:
        print(f"✅ {role} | Match: {score:.2f}%")
        print(f"   Required Skills: {matched_skills}\n")
    elif score == 0 and index == top_matches.index[0]:
        print("⚠️ No direct match found (Orthogonal). Try entering different tech skills.")
        break

print("-" * 50)
print("Matchmaking complete. Good luck on your career path!")