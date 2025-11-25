import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
from PIL import Image
import re
import tempfile
import shutil

def find_matching_images(original_dir, cropped_base_dir):
    """Find and match original images with their cropped versions"""
    
    matches = []
    
    # Get all original images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    original_files = [f for f in os.listdir(original_dir) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
    
    print(f"Found {len(original_files)} original images")
    
    # For each original image, find corresponding cropped images
    for original_file in sorted(original_files):
        original_path = os.path.join(original_dir, original_file)
        
        # Extract base name without extension
        base_name = os.path.splitext(original_file)[0]
        
        # Look for corresponding cropped images in set_image* directories
        set_dir = os.path.join(cropped_base_dir, f"set_{base_name}")
        
        rank_01_path = None
        rank_02_path = None
        rank_03_path = None
        
        if os.path.exists(set_dir):
            # Look for rank_01, rank_02, rank_03 images
            for file in os.listdir(set_dir):
                if "rank_01" in file and file.endswith(".png"):
                    rank_01_path = os.path.join(set_dir, file)
                elif "rank_02" in file and file.endswith(".png"):
                    rank_02_path = os.path.join(set_dir, file)
                elif "rank_03" in file and file.endswith(".png"):
                    rank_03_path = os.path.join(set_dir, file)
        
        matches.append({
            'name': base_name,
            'original_path': original_path,
            'rank_01_path': rank_01_path if rank_01_path and os.path.exists(rank_01_path) else None,
            'rank_02_path': rank_02_path if rank_02_path and os.path.exists(rank_02_path) else None,
            'rank_03_path': rank_03_path if rank_03_path and os.path.exists(rank_03_path) else None
        })
        
        status_01 = "✓" if rank_01_path else "✗"
        status_02 = "✓" if rank_02_path else "✗"
        status_03 = "✓" if rank_03_path else "✗"
        print(f"{base_name}: Rank01:{status_01} Rank02:{status_02} Rank03:{status_03}")
    
    return matches

def resize_image_for_excel(image_path, max_width=400, max_height=300, temp_dir=None):
    """Resize image to fit in Excel cell while maintaining aspect ratio"""
    try:
        img = Image.open(image_path)
        
        # Calculate scaling factor
        width_ratio = max_width / img.width
        height_ratio = max_height / img.height
        scale = min(width_ratio, height_ratio)
        
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        # Create resized image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save to temporary location in temp directory
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=temp_dir)
        temp_path = temp_file.name
        temp_file.close()
        
        img_resized.save(temp_path, quality=95)
        
        return temp_path, new_width, new_height
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")
        return None, 0, 0

def create_excel_with_images(matches, output_file):
    """Create Excel file with original and cropped images"""
    
    # Create temporary directory for resized images
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        # Create DataFrame with basic info
        df = pd.DataFrame({
            'Name': [m['name'] for m in matches],
            'Original Image': [''] * len(matches),
            'Rank 01': [''] * len(matches),
            'Rank 02': [''] * len(matches),
            'Rank 03': [''] * len(matches)
        })
        
        # Save initial DataFrame
        df.to_excel(output_file, index=False, engine='openpyxl')
        
        # Load workbook to add images
        wb = load_workbook(output_file)
        ws = wb.active
        
        # Set column widths
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 55
        ws.column_dimensions['C'].width = 55
        ws.column_dimensions['D'].width = 55
        ws.column_dimensions['E'].width = 55
        
        # Add images
        for idx, match in enumerate(matches, start=2):  # Start from row 2 (after header)
            row_num = idx
            
            # Set row height (in points, 1 point ≈ 1.33 pixels)
            ws.row_dimensions[row_num].height = 240
            
            # Add original image
            if match['original_path'] and os.path.exists(match['original_path']):
                try:
                    temp_path, width, height = resize_image_for_excel(match['original_path'], temp_dir=temp_dir)
                    if temp_path:
                        temp_files.append(temp_path)
                        img = OpenpyxlImage(temp_path)
                        img.width = width
                        img.height = height
                        ws.add_image(img, f'B{row_num}')
                except Exception as e:
                    print(f"Error adding original image for {match['name']}: {e}")
            
            # Add rank_01 image
            if match['rank_01_path'] and os.path.exists(match['rank_01_path']):
                try:
                    temp_path, width, height = resize_image_for_excel(match['rank_01_path'], temp_dir=temp_dir)
                    if temp_path:
                        temp_files.append(temp_path)
                        img = OpenpyxlImage(temp_path)
                        img.width = width
                        img.height = height
                        ws.add_image(img, f'C{row_num}')
                except Exception as e:
                    print(f"Error adding rank_01 image for {match['name']}: {e}")
            
            # Add rank_02 image
            if match['rank_02_path'] and os.path.exists(match['rank_02_path']):
                try:
                    temp_path, width, height = resize_image_for_excel(match['rank_02_path'], temp_dir=temp_dir)
                    if temp_path:
                        temp_files.append(temp_path)
                        img = OpenpyxlImage(temp_path)
                        img.width = width
                        img.height = height
                        ws.add_image(img, f'D{row_num}')
                except Exception as e:
                    print(f"Error adding rank_02 image for {match['name']}: {e}")
            
            # Add rank_03 image
            if match['rank_03_path'] and os.path.exists(match['rank_03_path']):
                try:
                    temp_path, width, height = resize_image_for_excel(match['rank_03_path'], temp_dir=temp_dir)
                    if temp_path:
                        temp_files.append(temp_path)
                        img = OpenpyxlImage(temp_path)
                        img.width = width
                        img.height = height
                        ws.add_image(img, f'E{row_num}')
                except Exception as e:
                    print(f"Error adding rank_03 image for {match['name']}: {e}")
            
            print(f"Added row {idx-1}/{len(matches)}: {match['name']}")
        
        # Save workbook (this reads all the temp files)
        wb.save(output_file)
        
        print(f"\n{'='*50}")
        print(f"Excel file created: {output_file}")
        print(f"Total rows: {len(matches)}")
        
    finally:
        # Clean up temporary directory and files
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")

if __name__ == "__main__":
    # Set your directories
    original_dir = "Input/SVTP803_Machu1416 CMIT_PV_CM01_picture"
    cropped_base_dir = "Top_Confident_Windows"
    output_file = "image_alignment.xlsx"
    
    # Find matching images
    print("Finding matching images...")
    matches = find_matching_images(original_dir, cropped_base_dir)
    
    # Create Excel file
    print("\nCreating Excel file with images...")
    create_excel_with_images(matches, output_file)
    
    print(f"\nComplete! Open {output_file} to view the results.")