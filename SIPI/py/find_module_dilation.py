import cv2
import numpy as np
import os
import fitz  # PyMuPDF
def count_and_visualize_contours(
    img_or_path,
    thresh=50,
    min_area=5000,
    kernel_size=11,
    iterations=5,
    black_tol=40,
    fill_holes=True,
    save_path=None,
    crop_contours=False,
    crop_base_dir=None,
    page_idx=None
):
    # --- accept path or already-loaded image ---
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {img_or_path}")
    else:
        img = img_or_path.copy()
        if img is None:
            raise ValueError("Input image array is None")

    # Store original image before any modifications for cropping
    original_img = img.copy()

    lower_black = np.array([0, 0, 0], np.uint8)
    upper_black = np.array([black_tol, black_tol, black_tol], np.uint8)
    mask_black = cv2.inRange(img, lower_black, upper_black)
    img[mask_black > 0] = (255, 255, 255)

    inv = 255 - img
    gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=iterations)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if fill_holes:
        filled = np.zeros_like(dilated)
        cv2.fillPoly(filled, pts=contours, color=255)
        contours2, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        work_mask = filled
        work_contours = contours2
    else:
        work_mask = dilated
        work_contours = contours

    large_contours = [cnt for cnt in work_contours if cv2.contourArea(cnt) > min_area]
    count = len(large_contours)

    vis = img.copy()
    cv2.drawContours(vis, large_contours, -1, (0, 255, 0), 2)
    
    # Crop and save individual contours if requested
    if crop_contours and crop_base_dir and page_idx is not None:
        page_dir = os.path.join(crop_base_dir, f"page_{page_idx:03d}")
        os.makedirs(page_dir, exist_ok=True)
    
    for i, cnt in enumerate(large_contours, 1):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        area = int(cv2.contourArea(cnt))
        cv2.putText(vis, f"#{i} area={area}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Crop and save the bounding box region
        if crop_contours and crop_base_dir and page_idx is not None:
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            x_end = min(original_img.shape[1], x + w)
            y_end = min(original_img.shape[0], y + h)
            
            # Use original_img instead of modified img to preserve text appearance
            cropped_img = original_img[y:y_end, x:x_end]
            
            # Create subdirectory for this object
            obj_dir = os.path.join(crop_base_dir, f"page_{page_idx:03d}", str(i))
            os.makedirs(obj_dir, exist_ok=True)
            
            # Save cropped image
            crop_path = os.path.join(obj_dir, f"obj_{i}.png")
            cv2.imwrite(crop_path, cropped_img)

    cv2.putText(vis, f"Count (area>{min_area}) = {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36, 255, 12), 2, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, vis)

    return count, vis, binary, work_mask

def pdf_pages_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    imgs = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        imgs.append(arr)
    doc.close()
    return imgs

def crop_inside(img, margin_ratio=0.02):
    h, w = img.shape[:2]
    margin_h = int(h * margin_ratio)
    margin_w = int(w * margin_ratio)
    return img[margin_h:h - margin_h, margin_w:w - margin_w]


def process_pdf(
    pdf_path,
    out_dir="pdf_pages_output",
    thresh=50,
    min_area=5000,
    kernel_size=11,
    iterations=3,
    black_tol=40,
    fill_holes=True,
    dpi=300,
    crop_contours=True,
    start_page=0,
    end_page=100
):
    os.makedirs(out_dir, exist_ok=True)
    
    # Create subdirectory for cropped objects
    crop_dir = os.path.join(out_dir, "cropped_objects")
    if crop_contours:
        os.makedirs(crop_dir, exist_ok=True)
    
    page_images = pdf_pages_to_images(pdf_path, dpi=dpi)
    results = []
    for idx, img in enumerate(page_images, start=1):
        if idx < start_page:
            continue
        
        img = crop_inside(img, margin_ratio=0.02)  # Crop 2% inward from each side
            
        base = f"page_{idx:03d}"
        vis_path = os.path.join(out_dir, f"{base}_vis.png")
        binary_path = os.path.join(out_dir, f"{base}_binary.png")
        mask_path = os.path.join(out_dir, f"{base}_mask.png")

        count, vis, binary, mask = count_and_visualize_contours(
            img,
            thresh=thresh,
            min_area=min_area,
            kernel_size=kernel_size,
            iterations=iterations,
            black_tol=black_tol,
            fill_holes=fill_holes,
            save_path=vis_path,
            crop_contours=crop_contours,
            crop_base_dir=crop_dir,
            page_idx=idx
        )
        cv2.imwrite(binary_path, binary)
        cv2.imwrite(mask_path, mask)

        results.append({
            "page": idx,
            "count": count,
            "vis_path": vis_path,
            "binary_path": binary_path,
            "mask_path": mask_path
        })
        print(f"Page {idx}: count={count}")
        if idx > end_page:
            break
    return results


# Example single image (still works)
# count, vis, binary, dilated = count_and_visualize_contours("temp.png")

# New PDF processing example
pdf_path = r"data/G2i_MerinoW1416_SI_0716 1.pdf"
summary = process_pdf(pdf_path, out_dir="pdf_pages_output", start_page=89, end_page=92)

print("Summary:", summary)