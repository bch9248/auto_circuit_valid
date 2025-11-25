import fitz  # PyMuPDF
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import json

class PDFTextExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_document = fitz.open(pdf_path)
        self.corpus = {}  # Dictionary to store text data for each page
        
    def extract_all_pages(self):
        """Extract text and positions from all pages of the PDF"""
        print(f"Extracting text from {len(self.pdf_document)} pages...")
        
        for page_num in range(len(self.pdf_document)):
            page = self.pdf_document[page_num]
            page_data = {
                'words': [],
                'text_blocks': [],
                'formatted_text': []
            }
            
            # Method 1: Extract words with positions
            words = page.get_text("words")  # Returns list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            for word in words:
                page_data['words'].append({
                    'text': word[4],
                    'bbox': word[:4],  # x0, y0, x1, y1
                    'block_no': word[5],
                    'line_no': word[6],
                    'word_no': word[7]
                })
            
            # Method 2: Extract text blocks
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                if len(block) > 4:  # Text block
                    page_data['text_blocks'].append({
                        'text': block[4],
                        'bbox': block[:4]
                    })
            
            # Method 3: Extract formatted text with spans
            text_dict = page.get_text("dict")
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_data['formatted_text'].append({
                                'text': span['text'],
                                'font': span['font'],
                                'size': span['size'],
                                'bbox': span['bbox']
                            })
            
            self.corpus[page_num] = page_data
            
            if (page_num + 1) % 10 == 0:
                print(f"Processed {page_num + 1} pages...")
        
        print(f"Extraction complete! Processed {len(self.pdf_document)} pages.")
        return self.corpus
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\#\-\_\%]', '', text)
        return text.upper()
    
    def get_page_text_variations(self, page_num: int) -> Set[str]:
        """Get all text variations from a page for matching"""
        if page_num not in self.corpus:
            return set()
        
        all_texts = set()
        page_data = self.corpus[page_num]
        
        # Add individual words
        for word in page_data['words']:
            all_texts.add(self.normalize_text(word['text']))
        
        # Add text from blocks (might contain multi-word strings)
        for block in page_data['text_blocks']:
            block_text = self.normalize_text(block['text'])
            all_texts.add(block_text)
            # Also add individual lines and words from blocks
            for line in block_text.split('\n'):
                all_texts.add(line.strip())
                for word in line.split():
                    all_texts.add(word.strip())
        
        # Add formatted text spans
        for span in page_data['formatted_text']:
            span_text = self.normalize_text(span['text'])
            all_texts.add(span_text)
        
        return {text for text in all_texts if text}  # Remove empty strings
    
    def calculate_page_confidence(self, page_num: int, target_strings: List[str]) -> Tuple[int, List[str]]:
        """Calculate confidence score for a page based on target strings"""
        page_texts = self.get_page_text_variations(page_num)
        normalized_targets = [self.normalize_text(target) for target in target_strings]
        
        matched_strings = []
        confidence_score = 0
        
        for i, target in enumerate(normalized_targets):
            # Exact match
            if target in page_texts:
                matched_strings.append(target_strings[i])  # Use original case
                confidence_score += 1
                continue
            
            # Partial match (target contains in any page text or vice versa)
            found_partial = False
            for page_text in page_texts:
                if target in page_text or page_text in target:
                    matched_strings.append(target_strings[i])
                    confidence_score += 0.5  # Partial match gets half score
                    found_partial = True
                    break
            
            if found_partial:
                continue
                
            # Fuzzy match for strings with spaces/special chars
            target_words = set(target.split())
            for page_text in page_texts:
                page_words = set(page_text.split())
                if target_words.intersection(page_words):
                    overlap_ratio = len(target_words.intersection(page_words)) / len(target_words)
                    if overlap_ratio >= 0.5:  # At least 50% word overlap
                        matched_strings.append(target_strings[i])
                        confidence_score += overlap_ratio * 0.3  # Fuzzy match gets proportional score
                        break
        
        return confidence_score, matched_strings
    
    def generate_page_scores_report(self, target_texts_sets: Dict[int, List[str]]) -> Dict:
        """Generate comprehensive JSON report with scores for all pages"""
        report = {
            "pdf_path": self.pdf_path,
            "total_pages": len(self.pdf_document),
            "target_sets": {},
            "page_scores": {}  # page_num -> {set_id: {score, matched_strings, match_percentage}}
        }
        
        # Initialize page_scores structure
        for page_num in range(len(self.pdf_document)):
            report["page_scores"][str(page_num)] = {}
        
        for set_id, target_strings in target_texts_sets.items():
            print(f"Generating report for target set {set_id}...")
            
            set_report = {
                "target_count": len(target_strings),
                "target_strings": target_strings,
                "pages_with_matches": [],
                "pages_without_matches": [],
                "statistics": {
                    "total_pages_with_matches": 0,
                    "total_pages_without_matches": 0,
                    "average_score": 0,
                    "max_score": 0,
                    "min_score": float('inf')
                }
            }
            
            all_scores = []
            
            for page_num in range(len(self.pdf_document)):
                confidence, matched = self.calculate_page_confidence(page_num, target_strings)
                match_percentage = (len(matched) / len(target_strings)) * 100 if target_strings else 0
                
                page_info = {
                    "page_number": page_num,
                    "confidence_score": round(confidence, 3),
                    "matched_count": len(matched),
                    "total_targets": len(target_strings),
                    "match_percentage": round(match_percentage, 2),
                    "matched_strings": matched
                }
                
                # Add to page_scores
                report["page_scores"][str(page_num)][f"set_{set_id}"] = {
                    "score": round(confidence, 3),
                    "matched_strings": matched,
                    "match_percentage": round(match_percentage, 2)
                }
                
                if confidence > 0:
                    set_report["pages_with_matches"].append(page_info)
                    all_scores.append(confidence)
                else:
                    set_report["pages_without_matches"].append(page_num)
            
            # Calculate statistics
            if all_scores:
                set_report["statistics"]["total_pages_with_matches"] = len(all_scores)
                set_report["statistics"]["total_pages_without_matches"] = len(self.pdf_document) - len(all_scores)
                set_report["statistics"]["average_score"] = round(sum(all_scores) / len(all_scores), 3)
                set_report["statistics"]["max_score"] = round(max(all_scores), 3)
                set_report["statistics"]["min_score"] = round(min(all_scores), 3)
            else:
                set_report["statistics"]["total_pages_without_matches"] = len(self.pdf_document)
                set_report["statistics"]["min_score"] = 0
            
            # Sort pages by confidence score
            set_report["pages_with_matches"].sort(key=lambda x: x["confidence_score"], reverse=True)
            
            report["target_sets"][str(set_id)] = set_report
        
        return report
    
    def save_scores_report(self, target_texts_sets: Dict[int, List[str]], filename: str = "page_scores_report.json"):
        """Generate and save comprehensive scores report to JSON"""
        report = self.generate_page_scores_report(target_texts_sets)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comprehensive scores report saved to {filename}")
        return report
    
    def find_best_pages(self, target_texts_sets: Dict[int, List[str]], top_k: int = 5) -> Dict[int, List[Tuple[int, float, List[str]]]]:
        """Find top K pages with highest confidence for each target set"""
        results = {}
        
        for set_id, target_strings in target_texts_sets.items():
            print(f"\nProcessing target set {set_id} with {len(target_strings)} strings...")
            
            page_scores = []
            for page_num in range(len(self.pdf_document)):
                confidence, matched = self.calculate_page_confidence(page_num, target_strings)
                if confidence > 0:  # Only include pages with some matches
                    page_scores.append((page_num, confidence, matched))
            
            # Sort by confidence score (descending)
            page_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top K pages
            results[set_id] = page_scores[:top_k]
            
            print(f"Found {len(page_scores)} pages with matches for set {set_id}")
            for i, (page_num, score, matched) in enumerate(results[set_id]):
                print(f"  {i+1}. Page {page_num}: Score {score:.2f}, Matched: {len(matched)} strings")
        
        return results
    
    def save_corpus(self, filename: str):
        """Save the corpus to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.corpus, f, indent=2)
        print(f"Corpus saved to {filename}")
    
    def load_corpus(self, filename: str):
        """Load corpus from a JSON file"""
        with open(filename, 'r') as f:
            self.corpus = json.load(f)
        print(f"Corpus loaded from {filename}")
    
    def close(self):
        """Close the PDF document"""
        self.pdf_document.close()
    def adjust_pagenum(self,num):
        # Since the corpus is 0-based indexing, need to compensate
        return num+1
def extract_page_png(pdf_document, extract_num=58, dpi=300):
    # Get page 87 (0-based indexing, so page 86)
    page = pdf_document[extract_num]
    
    # Maximum quality settings
    zoom = dpi / 72  # 600 DPI for very high quality
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(
        matrix=mat,
        alpha=False,  # No transparency for smaller file size
        colorspace=fitz.csRGB  # Ensure RGB color space
    )
    
    img_data = pix.tobytes("png")

    with open(f'page_{extract_num}_300dpi.png', 'wb') as f:
        f.write(img_data)

# Usage example
if __name__ == "__main__":
    # Initialize extractor
    input_file='data/G12_MACHU14_TLD_1217.pdf'
    extractor = PDFTextExtractor(input_file)
    pdf_document = fitz.open('data/G12_MACHU14_TLD_1217.pdf')
    
    # Extract text from all pages (this might take a while for large PDFs)
    corpus = extractor.extract_all_pages()
    
    # Save corpus for future use
    extractor.save_corpus('pdf_corpus.json')
    
    # Define your target text sets
    target_texts_sets = {
        6: [ # G12_MACHU14_TLD_1217 P 87
            "SIM", "CARD", "SOCKET", "&", "5G", "IN", "OUT", "WWAN_UIM_PWR", 
            "WWAN_UIM_RST", "WWAN_UIM_CLK", "WWAN_UIM_CD#", "UIM_VPP_CN", 
            "WWAN_UIM_DATA", "100K", "5%", "1", "0.41MM", "NT", "87", "20K", 
            "P3V3DS", "CATHODE", "ANODE", "BAV99", "COM", "C1451", "C1452", 
            "C1453", "C1416", "18PF", "50V", "0.1UF", "16V", "4.7UF", "6.3V", 
            "CAP", "CLOSE", "TO", "SIM", "CARD", "CN1400", "C1", "C2", "C3", 
            "C5", "C6", "C7", "VCC", "RST", "CLK", "GND", "CD/SW", "SW", 
            "VPP", "DATA", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", 
            "1.65MM", "0.75MM", "0.65MM", "0.43MM", "6026B0588101_TK2", 
            "FULLINHOPE_A1006_L001_7P"
        ]
    }
    
    # Generate and save comprehensive scores report
    scores_report = extractor.save_scores_report(target_texts_sets, "page_scores_report.json")
    
    # Find best matching pages
    results = extractor.find_best_pages(target_texts_sets, top_k=2)
    
    # Display detailed results
    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    show_page_threshold=2
    for set_id, page_results in results.items():
        
        print(f"\nTarget Set {set_id}:")
        print("-" * 30)
        
        for rank, (page_num, score, matched_strings) in enumerate(page_results, 1):
            print(f"\nRank {rank}: Page {page_num} (Score: {score:.2f})")
            print(f"Matched {len(matched_strings)} out of {len(target_texts_sets[set_id])} strings:")
            for matched in matched_strings[:10]:  # Show first 10 matches
                print(f"  ✓ {matched}")
            if len(matched_strings) > 10:
                print(f"  ... and {len(matched_strings) - 10} more")

            if rank <= show_page_threshold:
                # page_num_adjusted=extractor.adjust_pagenum(page_num)
                extract_page_png(pdf_document, extract_num=page_num)
    # Close the PDF
    extractor.close()
    pdf_document.close()