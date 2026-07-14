import urllib.request
import urllib.parse
import json
from src.utils import setup_logging

logger = setup_logging("translator")


def translate_text_free(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using free Google Translate API."""
    if not text.strip():
        return text

    # Map lang codes for Google Translate
    # e.g., 'en-US' -> 'en', 'zh-CN' -> 'zh-CN', 'vi-VN' -> 'vi'
    sl = source_lang.split("-")[0]
    tl = target_lang.split("-")[0]

    try:
        url = (
            f"https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={sl}&tl={tl}&dt=t&q={urllib.parse.quote(text)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            translated = "".join([part[0] for part in data[0] if part[0]])
            return translated
    except Exception as e:
        logger.error(f"Free translation failed: {e}")
        # Return original text as fallback if translation fails
        return text


def translate_segments_vi(segments: list[dict], source_lang: str) -> list[dict]:
    """Translate the 'text' field of each segment to Vietnamese and add 'text_vi'."""
    logger.info(f"Translating {len(segments)} segments to Vietnamese...")
    
    # We can batch the translations to be faster and respect limits
    # Join with a unique delimiter that Google Translate preserves
    delimiter = "\n---\n"
    texts = [seg["text"] for seg in segments]
    
    # Group into batches of ~15 segments to avoid URL length limit (approx 2000 chars)
    batch_size = 15
    translated_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        combined = delimiter.join(batch)
        translated_combined = translate_text_free(combined, source_lang, "vi-VN")
        
        # Split back
        parts = translated_combined.split(delimiter)
        # Clean up parts (sometimes Google Translate modifies spaces around delimiter)
        parts = [p.strip() for p in parts]
        
        # If split count matches, append them, otherwise fallback to item-by-item for this batch
        if len(parts) == len(batch):
            translated_texts.extend(parts)
        else:
            logger.warning("Batch split mismatch, falling back to segment-by-segment translation for this batch")
            for item in batch:
                translated_texts.append(translate_text_free(item, source_lang, "vi-VN"))

    # Assign translations
    for seg, trans in zip(segments, translated_texts):
        seg["text_vi"] = trans

    return segments


def translate_segments_jp(segments: list[dict], source_lang: str) -> list[dict]:
    """Translate the 'text' field of each segment to Japanese and add 'text_jp'."""
    logger.info(f"Translating {len(segments)} segments to Japanese...")
    
    delimiter = "\n---\n"
    texts = [seg["text"] for seg in segments]
    
    batch_size = 15
    translated_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        combined = delimiter.join(batch)
        translated_combined = translate_text_free(combined, source_lang, "ja-JP")
        
        parts = translated_combined.split(delimiter)
        parts = [p.strip() for p in parts]
        
        if len(parts) == len(batch):
            translated_texts.extend(parts)
        else:
            logger.warning("Batch split mismatch, falling back to segment-by-segment translation for this batch")
            for item in batch:
                translated_texts.append(translate_text_free(item, source_lang, "ja-JP"))

    for seg, trans in zip(segments, translated_texts):
        seg["text_jp"] = trans

    return segments
