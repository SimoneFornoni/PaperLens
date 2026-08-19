import re

def sanitize_model_name(model_name: str) -> str:
    """Convvert model name to safe collection or file name
    
    Args:
        model_name (str): The model name to sanitize.
        
    Returns:
        str: Sanitized model name suitable for collection or file naming.
    """
    return re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)

def sanitize_filename(text: str) -> str:
    """Removes characters that are not allowed in file systems."""
    return re.sub(r'[\\/*?:"<>|]', "", text).strip()