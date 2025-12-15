"""
Script to download NLTK data during Docker build.
Handles SSL errors gracefully in CI/CD environments.
"""
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: NLTK data download failed: {e}')
    print('NLTK will attempt to download data on first run')
