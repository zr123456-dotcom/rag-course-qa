from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love Python",
    "Python is great",
    "I love machine learning"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

print(tfidf_matrix.shape)          # (3, 6) 3篇文档，6个词
print(tfidf_matrix.toarray())
print(vectorizer.get_feature_names_out())
print()
print('='*10)
from sklearn.metrics.pairwise import cosine_similarity

# 两个向量
a = [[1, 2, 3]]
b = [[1, 2, 4]]

print(cosine_similarity(a, b))  # [[0.99...]]

# 多篇文档的 TF-IDF 矩阵，计算两两相似度
similarities = cosine_similarity(tfidf_matrix)
print(similarities)