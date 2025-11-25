import fitz  # PyMuPDF
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import json
from PIL import Image
import os
import numpy as np

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
    
    def calculate_page_confidence_with_coords(self, page_num: int, target_strings: List[str]) -> Tuple[float, List[str], List[Dict]]:
        """Calculate confidence score and return matched text coordinates"""
        if page_num not in self.corpus:
            return 0, [], []
        
        page_data = self.corpus[page_num]
        normalized_targets = [self.normalize_text(target) for target in target_strings]
        
        matched_strings = []
        matched_coords = []  # Store coordinates of matched texts
        confidence_score = 0
        
        # Create a lookup for all text elements with coordinates
        text_elements = []
        
        # Add words with coordinates
        for word in page_data['words']:
            text_elements.append({
                'text': word['text'],
                'normalized': self.normalize_text(word['text']),
                'bbox': word['bbox'],
                'type': 'word'
            })
        
        # Add formatted spans with coordinates
        for span in page_data['formatted_text']:
            text_elements.append({
                'text': span['text'],
                'normalized': self.normalize_text(span['text']),
                'bbox': span['bbox'],
                'type': 'span'
            })
        
        for i, target in enumerate(normalized_targets):
            matched = False
            
            # Check for exact matches
            for element in text_elements:
                if target == element['normalized']:
                    matched_strings.append(target_strings[i])
                    matched_coords.append({
                        'original_text': target_strings[i],
                        'matched_text': element['text'],
                        'bbox': element['bbox'],
                        'match_type': 'exact',
                        'score': 1.0
                    })
                    confidence_score += 1
                    matched = True
                    break
            
            if matched:
                continue
            
            # Check for partial matches
            for element in text_elements:
                if target in element['normalized'] or element['normalized'] in target:
                    matched_strings.append(target_strings[i])
                    matched_coords.append({
                        'original_text': target_strings[i],
                        'matched_text': element['text'],
                        'bbox': element['bbox'],
                        'match_type': 'partial',
                        'score': 0.5
                    })
                    confidence_score += 0.5
                    matched = True
                    break
            
            if matched:
                continue
            
            # Fuzzy match for multi-word strings
            target_words = set(target.split())
            for element in text_elements:
                element_words = set(element['normalized'].split())
                if target_words.intersection(element_words):
                    overlap_ratio = len(target_words.intersection(element_words)) / len(target_words)
                    if overlap_ratio >= 0.5:
                        matched_strings.append(target_strings[i])
                        matched_coords.append({
                            'original_text': target_strings[i],
                            'matched_text': element['text'],
                            'bbox': element['bbox'],
                            'match_type': 'fuzzy',
                            'score': overlap_ratio * 0.3
                        })
                        confidence_score += overlap_ratio * 0.3
                        break
        
        return confidence_score, matched_strings, matched_coords
    
    def sliding_window_analysis(self, page_num: int, target_strings: List[str], 
                           window_size=(1400, 500), overlap=0.3, dpi=300):
        """Analyze page using sliding window to find best regions"""
        print(f"Analyzing page {page_num} with sliding window...")
        
        # Get page dimensions
        page = self.pdf_document[page_num]
        page_rect = page.rect
        zoom = dpi / 72
        
        # Generate and save full page image with coordinates marked
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
        img_width = pix.width
        img_height = pix.height
        
        print(f"Page dimensions: {img_width} x {img_height} pixels")
        
        # Save base page image
        page_img_path = f'page_{page_num}_{dpi}dpi_base.png'
        pix.save(page_img_path)
        
        # Get matched coordinates
        confidence, matched_strings, matched_coords = self.calculate_page_confidence_with_coords(page_num, target_strings)
        
        if not matched_coords:
            print(f"No matches found on page {page_num}")
            return []
        
        print(f"Found {len(matched_coords)} matches")
        
        # Convert PDF coordinates to image coordinates
        image_coords = []
        for match in matched_coords:
            pdf_bbox = match['bbox']  # x0, y0, x1, y1 in PDF coordinates
            
            # Convert PDF coordinates to image pixel coordinates
            img_x0 = int(pdf_bbox[0] * zoom)
            img_y0 = int(pdf_bbox[1] * zoom)
            img_x1 = int(pdf_bbox[2] * zoom)
            img_y1 = int(pdf_bbox[3] * zoom)
            
            # Calculate center point
            center_x = (img_x0 + img_x1) // 2
            center_y = (img_y0 + img_y1) // 2
            
            image_coords.append({
                'original_text': match['original_text'],
                'matched_text': match['matched_text'],
                'bbox': (img_x0, img_y0, img_x1, img_y1),
                'center': (center_x, center_y),
                'match_type': match['match_type'],
                'score': match['score']
            })
            
            print(f"'{match['original_text']}' -> IMG({img_x0},{img_y0},{img_x1},{img_y1})")
        
        # Generate sliding windows
        window_width, window_height = window_size
        step_x = int(window_width * (1 - overlap))
        step_y = int(window_height * (1 - overlap))
        
        windows = []
        window_id = 0
        
        # Generate window positions
        x_positions = list(range(0, max(1, img_width - window_width + 1), step_x))
        if not x_positions or x_positions[-1] + window_width < img_width:
            x_positions.append(max(0, img_width - window_width))
        
        y_positions = list(range(0, max(1, img_height - window_height + 1), step_y))
        if not y_positions or y_positions[-1] + window_height < img_height:
            y_positions.append(max(0, img_height - window_height))
        
        for y in y_positions:
            for x in x_positions:
                # Ensure window doesn't exceed boundaries
                actual_x = max(0, min(x, img_width - window_width))
                actual_y = max(0, min(y, img_height - window_height))
                
                window_bbox = (actual_x, actual_y, 
                            actual_x + window_width, 
                            actual_y + window_height)
                
                # Count matches in this window
                matches_in_window = []
                window_score = 0
                
                for coord in image_coords:
                    text_bbox = coord['bbox']
                    text_center = coord['center']
                    
                    # Check if text center is within window
                    if (window_bbox[0] <= text_center[0] <= window_bbox[2] and 
                        window_bbox[1] <= text_center[1] <= window_bbox[3]):
                        matches_in_window.append(coord)
                        window_score += coord['score']
                
                if matches_in_window:
                    window_info = {
                        'window_id': window_id,
                        'bbox': window_bbox,
                        'four_corners': [
                            (actual_x, actual_y),
                            (actual_x + window_width, actual_y),
                            (actual_x + window_width, actual_y + window_height),
                            (actual_x, actual_y + window_height)
                        ],
                        'matches': matches_in_window,
                        'match_count': len(matches_in_window),
                        'window_score': window_score,
                        'coverage_ratio': len(matches_in_window) / len(target_strings)
                    }
                    windows.append(window_info)
                
                window_id += 1
        
        # Sort windows by score
        windows.sort(key=lambda w: (w['window_score'], w['match_count']), reverse=True)
        
        print(f"Generated {len(windows)} windows with matches")
        return windows
    
    def crop_best_windows(self, page_num: int, target_strings: List[str], 
                         top_k_windows=3, window_size=(1400, 500), overlap=0.3, dpi=300):
        """Find best windows and crop them from the page image"""
        
        # Generate page image
        page = self.pdf_document[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
        
        # Save full page image
        img_filename = f'page_{page_num}_{dpi}dpi.png'
        pix.save(img_filename)
        
        # Analyze with sliding window
        windows = self.sliding_window_analysis(page_num, target_strings, window_size, overlap, dpi)
        
        if not windows:
            print(f"No windows with matches found on page {page_num}")
            return []
        
        # Take top K windows
        best_windows = windows[:top_k_windows]
        
        # Create output directory
        output_dir = f"cropped_windows_page_{page_num}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the full image for cropping
        full_image = Image.open(img_filename)
        
        cropped_results = []
        
        for i, window in enumerate(best_windows):
            # Crop the window
            crop_box = window['bbox']  # (x1, y1, x2, y2)
            cropped_img = full_image.crop(crop_box)
            
            # Save cropped window
            crop_filename = f"window_{i+1}_score_{window['window_score']:.2f}_matches_{window['match_count']}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            cropped_img.save(crop_path)
            
            # Create detailed report for this window
            window_report = {
                'rank': i + 1,
                'crop_path': crop_path,
                'window_id': window['window_id'],
                'bbox': window['bbox'],
                'four_corners': window['four_corners'],
                'match_count': window['match_count'],
                'window_score': window['window_score'],
                'coverage_ratio': window['coverage_ratio'],
                'matched_texts': [match['original_text'] for match in window['matches']],
                'match_details': window['matches']
            }
            
            cropped_results.append(window_report)
            
            print(f"Window {i+1}: Score {window['window_score']:.2f}, "
                  f"Matches: {window['match_count']}, "
                  f"Coverage: {window['coverage_ratio']:.2%}")
            print(f"  Saved to: {crop_path}")
            print(f"  Matched texts: {', '.join([m['original_text'] for m in window['matches']])}")
        
        # Save window analysis report
        report_path = os.path.join(output_dir, f"window_analysis_report_page_{page_num}.json")
        with open(report_path, 'w') as f:
            json.dump({
                'page_number': page_num,
                'total_windows_analyzed': len(windows),
                'top_windows_selected': len(best_windows),
                'window_size': window_size,
                'overlap': overlap,
                'dpi': dpi,
                'target_strings_count': len(target_strings),
                'windows': cropped_results
            }, f, indent=2)
        
        print(f"Window analysis report saved to: {report_path}")
        return cropped_results
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
    def get_top_windows_across_pages(self, target_texts_sets: Dict[int, List[str]], 
                                   top_pages: int = 2, top_windows_per_page: int = 5,
                                   final_top_k: int = 5, window_size=(1400, 500), 
                                   overlap=0.3, dpi=300):
        """
        Get top K windows across all best pages for each target set
        
        Args:
            target_texts_sets: Dictionary of target text sets
            top_pages: Number of top pages to analyze per set
            top_windows_per_page: Maximum windows to extract per page
            final_top_k: Final number of top windows to output
            window_size: Size of sliding window
            overlap: Overlap ratio between windows
            dpi: Image resolution
        
        Returns:
            Dictionary with top windows across all pages
        """
        
        # First, find best pages for each set
        best_pages_results = self.find_best_pages(target_texts_sets, top_k=top_pages)
        
        all_results = {}
        
        for set_id, target_strings in target_texts_sets.items():
            print(f"\n{'='*60}")
            print(f"PROCESSING TARGET SET {set_id}")
            print(f"{'='*60}")
            
            page_results = best_pages_results.get(set_id, [])
            
            if not page_results:
                print(f"No pages found for target set {set_id}")
                continue
            
            # Collect all windows from all best pages
            all_windows = []
            
            for rank, (page_num, page_score, matched_strings) in enumerate(page_results, 1):
                print(f"\nAnalyzing Page {page_num} (Rank {rank}, Score: {page_score:.2f})")
                
                # Get windows for this page
                windows = self.sliding_window_analysis(
                    page_num=page_num,
                    target_strings=target_strings,
                    window_size=window_size,
                    overlap=overlap,
                    dpi=dpi
                )
                
                # Add page info to each window
                for window in windows[:top_windows_per_page]:
                    window['page_num'] = page_num
                    window['page_rank'] = rank
                    window['page_score'] = page_score
                    # Calculate combined score (window score + page score factor)
                    window['combined_score'] = window['window_score'] + (page_score * 0.1)
                    all_windows.append(window)
                
                print(f"  Found {len(windows)} windows on page {page_num}")
            
            # Sort all windows by combined score
            all_windows.sort(key=lambda w: (w['combined_score'], w['window_score']), reverse=True)
            
            # Take top K windows
            top_windows = all_windows[:final_top_k]
            
            print(f"\nTop {final_top_k} windows across all pages for set {set_id}:")
            for i, window in enumerate(top_windows, 1):
                print(f"  {i}. Page {window['page_num']}, Window {window['window_id']}: "
                      f"Score {window['window_score']:.2f}, Combined {window['combined_score']:.2f}, "
                      f"Matches: {window['match_count']}")
            
            all_results[set_id] = {
                'target_strings': target_strings,
                'top_windows': top_windows,
                'total_windows_analyzed': len(all_windows),
                'pages_analyzed': len(page_results)
            }
        
        return all_results
    
    def visualize_windows_on_page(self, page_num: int, windows: List[Dict], 
                                 output_path: str = None, top_k: int = 10):
        """Draw windows on page image for visualization"""
        from PIL import ImageDraw, ImageFont
        
        # Load the base page image
        base_img_path = f'page_{page_num}_300dpi_base.png'
        if not os.path.exists(base_img_path):
            print(f"Base image not found: {base_img_path}")
            return
        
        img = Image.open(base_img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Colors for different ranks
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
        
        # Draw top K windows
        for i, window in enumerate(windows[:top_k]):
            bbox = window['bbox']
            color = colors[i % len(colors)]
            
            # Draw rectangle
            draw.rectangle(bbox, outline=color, width=5)
            
            # Draw label
            label = f"#{i+1} ({window['window_score']:.1f})"
            draw.text((bbox[0] + 10, bbox[1] + 10), label, fill=color)
            
            # Draw match points
            for match in window['matches']:
                center = match['center']
                # Draw small circle at text center
                r = 3
                draw.ellipse([center[0]-r, center[1]-r, center[0]+r, center[1]+r], 
                           fill=color, outline=color)
        
        # Save visualization
        if output_path is None:
            output_path = f'page_{page_num}_windows_visualization.png'
        
        img.save(output_path)
        print(f"Window visualization saved to: {output_path}")
        return output_path
    
    def crop_and_save_top_windows(self, target_texts_sets: Dict[int, List[str]], 
                                 top_pages: int = 2, top_windows_per_page: int = 5,
                                 final_top_k: int = 5, window_size=(1400, 500), 
                                 overlap=0.3, dpi=300, output_dir="Top_Confident_Windows"):
        """Extract and save the top K most confident windows with visualization"""
        
        all_results = self.get_top_windows_across_pages(
            target_texts_sets, top_pages, top_windows_per_page, 
            final_top_k, window_size, overlap, dpi
        )
        
        os.makedirs(output_dir, exist_ok=True)
        final_results = {}
        
        for set_id, results in all_results.items():
            print(f"\n{'='*50}")
            print(f"CROPPING TOP WINDOWS FOR SET {set_id}")
            print(f"{'='*50}")
            
            top_windows = results['top_windows']
            
            set_output_dir = os.path.join(output_dir, f"set_{set_id}")
            os.makedirs(set_output_dir, exist_ok=True)
            
            cropped_windows = []
            
            # Group windows by page for visualization
            page_windows = {}
            for window in top_windows:
                page_num = window['page_num']
                if page_num not in page_windows:
                    page_windows[page_num] = []
                page_windows[page_num].append(window)
            
            # Create visualization for each page
            for page_num, windows in page_windows.items():
                vis_path = os.path.join(set_output_dir, f"page_{page_num}_windows_marked.png")
                self.visualize_windows_on_page(page_num, windows, vis_path, top_k=len(windows))
            
            # Crop individual windows
            for i, window in enumerate(top_windows, 1):
                page_num = window['page_num']
                
                # Load full page image
                page_img_path = f'page_{page_num}_300dpi_base.png'
                full_image = Image.open(page_img_path)
                
                # Crop window
                crop_box = window['bbox']
                cropped_img = full_image.crop(crop_box)
                
                # Save cropped image
                crop_filename = (f"rank_{i:02d}_page_{page_num}_window_{window['window_id']}_"
                               f"score_{window['window_score']:.2f}.png")
                crop_path = os.path.join(set_output_dir, crop_filename)
                cropped_img.save(crop_path)
                
                window_report = {
                    'global_rank': i,
                    'page_num': page_num,
                    'window_id': window['window_id'],
                    'window_score': window['window_score'],
                    'match_count': window['match_count'],
                    'bbox': window['bbox'],
                    'crop_path': crop_path,
                    'matched_texts': [match['original_text'] for match in window['matches']]
                }
                
                cropped_windows.append(window_report)
                print(f"  ✓ Window {i} saved: {crop_filename}")
            
            # Save report
            report_data = {
                'set_id': set_id,
                'cropped_windows': cropped_windows
            }
            
            report_path = os.path.join(set_output_dir, f"windows_report_set_{set_id}.json")
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            final_results[set_id] = report_data
            print(f"  📋 Report saved: {report_path}")
        
        return final_results
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
    input_file='Input/G12_MACHU14_TLD_1217/G12_MACHU14_TLD_1217.pdf'
    extractor = PDFTextExtractor(input_file)
    pdf_document = fitz.open('Input/G12_MACHU14_TLD_1217/G12_MACHU14_TLD_1217.pdf')
    
    # Extract text from all pages (this might take a while for large PDFs)
    corpus = extractor.extract_all_pages()
    
    # Save corpus for future use
    extractor.save_corpus('pdf_corpus.json')

    with open('corpus/extracted_target_texts.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    target_texts_sets = data['target_texts_sets']
    # # Define your target text sets
    # target_texts_sets = {
    #     6: [ # G12_MACHU14_TLD_1217 P 87
    #         "SIM", "CARD", "SOCKET", "&", "5G", "IN", "OUT", "WWAN_UIM_PWR", 
    #         "WWAN_UIM_RST", "WWAN_UIM_CLK", "WWAN_UIM_CD#", "UIM_VPP_CN", 
    #         "WWAN_UIM_DATA", "100K", "5%", "1", "0.41MM", "NT", "87", "20K", 
    #         "P3V3DS", "CATHODE", "ANODE", "BAV99", "COM", "C1451", "C1452", 
    #         "C1453", "C1416", "18PF", "50V", "0.1UF", "16V", "4.7UF", "6.3V", 
    #         "CAP", "CLOSE", "TO", "SIM", "CARD", "CN1400", "C1", "C2", "C3", 
    #         "C5", "C6", "C7", "VCC", "RST", "CLK", "GND", "CD/SW", "SW", 
    #         "VPP", "DATA", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", 
    #         "1.65MM", "0.75MM", "0.65MM", "0.43MM", "6026B0588101_TK2", 
    #         "FULLINHOPE_A1006_L001_7P"
    #     ]
    # }
    
    # Generate and save comprehensive scores report
    scores_report = extractor.save_scores_report(target_texts_sets, "page_scores_report.json")
    
    # Find best matching pages
    results = extractor.find_best_pages(target_texts_sets, top_k=2)
    
    # Display detailed results and perform sliding window analysis
    print("\n" + "="*50)
    print("DETAILED RESULTS WITH SLIDING WINDOW ANALYSIS")
    print("="*50)
    
    window_size = (1400, 500)  # Window dimensions
    overlap = 0.3  # 30% overlap
    top_k_windows = 3  # Number of best windows to extract per page
    
    # Extract top 5 most confident windows across all best pages
    print("\n" + "="*60)
    print("EXTRACTING TOP 5 CONFIDENT WINDOWS ACROSS ALL PAGES")
    print("="*60)
    
    final_results = extractor.crop_and_save_top_windows(
        target_texts_sets=target_texts_sets,
        top_pages=2,  # Analyze top 2 pages per set
        top_windows_per_page=5,  # Max 5 windows per page
        final_top_k=5,  # Final top 5 windows
        window_size=window_size,
        overlap=0.3,
        dpi=300,
        output_dir="Top_Confident_Windows"
    )
    
    # Close the PDF
    extractor.close()
    pdf_document.close()
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print("Check the 'cropped_windows_page_X' directories for extracted window images")
    print("Check the JSON reports for detailed analysis results")