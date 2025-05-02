"""
Text Preprocessing Module - Prepares raw text documents for further processing.
"""
def clean_text(text: str) -> str:
    """
    Cleans input text by removing unwanted characters.

    :param text: Raw text input.
    :type text: str
    :return: Cleaned text.
    :rtype: str
    """
    return text.strip().replace("\n", " ").lower()
