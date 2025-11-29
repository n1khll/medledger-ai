"""PDF text extraction utility for medical records"""
import io
from typing import Optional
from shared.logging_config import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes.
    
    Tries multiple PDF extraction libraries in order:
    1. pdfplumber (better for structured documents)
    2. PyPDF2 (fallback)
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        Extracted text as string
        
    Raises:
        ValueError: If PDF extraction fails
    """
    text = None
    
    # Try pdfplumber first (better for medical records with tables)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
            text = "\n".join(full_text)
            if text and text.strip():
                logger.info(f"Successfully extracted {len(text)} characters using pdfplumber")
                return text.strip()
    except ImportError:
        logger.warning("pdfplumber not available, trying PyPDF2")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {str(e)}, trying PyPDF2")
    
    # Fallback to PyPDF2
    try:
        import PyPDF2
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
        text = "\n".join(full_text)
        if text and text.strip():
            logger.info(f"Successfully extracted {len(text)} characters using PyPDF2")
            return text.strip()
    except ImportError:
        logger.error("Neither pdfplumber nor PyPDF2 available. Please install: pip install pdfplumber PyPDF2")
        raise ValueError("PDF extraction libraries not available. Install: pip install pdfplumber PyPDF2")
    except Exception as e:
        logger.error(f"PyPDF2 extraction failed: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    # If we get here, extraction failed
    if not text or not text.strip():
        raise ValueError("No text could be extracted from PDF. The PDF may be image-based or corrupted.")
    
    return text.strip()

