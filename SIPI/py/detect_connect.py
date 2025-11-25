# Read and find if lines are connected in schemetic plot 

import cv2
import numpy as np
import os
from pathlib import Path
import math
from collections import defaultdict




EPS = 2.0  # pixels tolerance

class DSU:
    def __init__(self):
        self.p = {}
    def find(self, x):
        if x not in self.p:
            self.p[x] = x
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a,b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

def snap(v):  # optional grid snapping
    return round(v)

def norm_point(x,y):
    return (snap(x), snap(y))

def build_connectivity(segments, junction_points, texts):
    # segments: list of (id, x1,y1,x2,y2)
    # junction_points: list of (x,y)
    # texts: list of {'text': str, 'bbox': (x1,y1,x2,y2)}
    dsu = DSU()

    # 1. Register endpoints
    for sid,x1,y1,x2,y2 in segments:
        p1 = norm_point(x1,y1)
        p2 = norm_point(x2,y2)
        dsu.union(p1,p2)  # wire continuity

    # 2. Merge touching endpoints
    endpoints = [norm_point(s[1],s[2]) for s in segments] + [norm_point(s[3],s[4]) for s in segments]
    endpoint_set = set(endpoints)

    # 3. Junction dots: force union of all points within EPS
    for jx,jy in junction_points:
        jp = norm_point(jx,jy)
        for ex,ey in endpoint_set:
            if math.hypot(ex-jp[0], ey-jp[1]) <= EPS:
                dsu.union(jp,(ex,ey))

    # 4. Collinear tee detection (endpoint on segment interior)
    def on_segment(px,py,x1,y1,x2,y2):
        if x1 == x2:  # vertical
            if abs(px - x1) <= EPS and min(y1,y2)-EPS <= py <= max(y1,y2)+EPS:
                return True
        if y1 == y2:  # horizontal
            if abs(py - y1) <= EPS and min(x1,x2)-EPS <= px <= max(x1,x2)+EPS:
                return True
        return False

    for sid,x1,y1,x2,y2 in segments:
        for ep in list(endpoint_set):
            if ep != norm_point(x1,y1) and ep != norm_point(x2,y2):
                if on_segment(ep[0], ep[1], x1,y1,x2,y2):
                    dsu.union(ep, norm_point(x1,y1))

    # 5. Attach net labels
    net_map = defaultdict(set)  # net_name -> set(representatives)
    rep_name = {}  # representative -> net_name
    for t in texts:
        name = t['text'].strip()
        if not name or len(name) < 2:
            continue
        x1,y1,x2,y2 = t['bbox']
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2
        # find nearest endpoint
        if endpoint_set:
            nearest = min(endpoint_set, key=lambda p: math.hypot(p[0]-cx, p[1]-cy))
            if math.hypot(nearest[0]-cx, nearest[1]-cy) <= 50:  # heuristic distance
                r = dsu.find(nearest)
                net_map[name].add(r)

    # 6. Union aliases with identical text
    for name, reps in net_map.items():
        reps = list(reps)
        for i in range(1, len(reps)):
            dsu.union(reps[0], reps[i])

    # Refresh representative -> name
    for name, reps in net_map.items():
        for r in reps:
            rep_name[dsu.find(r)] = name

    return dsu, rep_name

def are_connected(dsu, rep_name, point_or_text_a, point_or_text_b):
    # Resolve to representative (point coordinates or text label)
    def resolve(e):
        if isinstance(e, tuple):  # coordinate
            return dsu.find(norm_point(*e))
        # assume text label string
        for r,name in rep_name.items():
            if name == e:
                return dsu.find(r)
        return None
    ra = resolve(point_or_text_a)
    rb = resolve(point_or_text_b)
    return ra is not None and rb is not None and dsu.find(ra) == dsu.find(rb)

def simple_ocr_analysis(image_path):
    """
    Simple OCR analysis if deepdoctection is too complex
    """
    try:
        import pytesseract
        import cv2
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold for better OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        
        # Process OCR data into text objects for connectivity analysis
        texts = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60 and data['text'][i].strip():  # Only confident detections
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                texts.append({
                    'text': data['text'][i].strip(),
                    'bbox': (x, y, x + w, y + h),
                    'confidence': data['conf'][i]
                })
        
        # Draw bounding boxes around detected text
        vis_img = img.copy()
        for text_obj in texts:
            x1, y1, x2, y2 = text_obj['bbox']
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, text_obj['text'], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save visualization
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_ocr.png"
        cv2.imwrite(str(output_path), vis_img)
        print(f"OCR visualization saved to: {output_path}")
        
        # Save OCR text results to file
        text_output_path = Path(image_path).parent / f"{Path(image_path).stem}_ocr_results.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results for: {Path(image_path).name}\n")
            f.write("=" * 50 + "\n\n")
            
            if texts:
                f.write(f"Total detected text objects: {len(texts)}\n\n")
                for i, text_obj in enumerate(texts, 1):
                    f.write(f"Text {i}:\n")
                    f.write(f"  Content: '{text_obj['text']}'\n")
                    f.write(f"  Confidence: {text_obj['confidence']}%\n")
                    f.write(f"  Bounding Box: {text_obj['bbox']}\n")
                    f.write(f"  Position: x={text_obj['bbox'][0]}, y={text_obj['bbox'][1]}\n")
                    f.write("\n")
                
                # Summary of all detected text
                f.write("Summary - All detected text:\n")
                f.write("-" * 30 + "\n")
                for text_obj in texts:
                    f.write(f"'{text_obj['text']}'\n")
            else:
                f.write("No text detected with sufficient confidence.\n")
        
        print(f"OCR text results saved to: {text_output_path}")
        
        return texts
        
    except ImportError:
        print("Please install pytesseract: pip install pytesseract")
        return None, []

def process_image_for_lines_and_nodes(image_path, output_path):
    """
    Process a single image to detect lines and circular nodes.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save processed image
    
    Returns:
        tuple: (success, lines, nodes) where lines is list of (id, x1,y1,x2,y2) and nodes is list of (x,y)
    """
    # Load the image
    if not os.path.exists(image_path):
        print(f"檔案不存在：{image_path}")
        return False, [], []

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Failed to load image from {image_path}. Please check the file path and integrity.")
        return False, [], []

    # Preprocess the image (edge detection)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)

    # Detect nodes (circular shapes) using Hough Circle Transform
    nodes = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=20)

    # Create output image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Process lines for connectivity analysis
    line_segments = []
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            line_segments.append((i, x1, y1, x2, y2))
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Process nodes for connectivity analysis
    junction_points = []
    if nodes is not None:
        nodes = np.uint16(np.around(nodes))
        for node in nodes[0, :]:
            x, y, radius = node
            junction_points.append((x, y))
            cv2.circle(output_image, (x, y), radius, (255, 0, 0), 2)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed image
    cv2.imwrite(output_path, output_image)
    return True, line_segments, junction_points

def analyze_connectivity(image_path):
    """
    Complete connectivity analysis combining line detection, OCR, and connectivity checking
    """
    print(f"Analyzing connectivity for: {image_path}")
    
    # Extract lines and nodes
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_line_extract.png"
    success, line_segments, junction_points = process_image_for_lines_and_nodes(image_path, str(output_path))
    
    if not success:
        return None
    
    # Extract text
    text_objects = simple_ocr_analysis(image_path)
    
    if not text_objects:
        print("No text detected")
        return None
    
    # Build connectivity graph
    dsu, rep_name = build_connectivity(line_segments, junction_points, text_objects)
    
    # Print detected entities
    print(f"\nDetected {len(line_segments)} line segments")
    print(f"Detected {len(junction_points)} junction points")
    print(f"Detected {len(text_objects)} text objects")
    
    # Print all detected net names
    net_names = list(set(rep_name.values()))
    print(f"\nDetected net names: {net_names}")
    
    return {
        'dsu': dsu,
        'rep_name': rep_name,
        'line_segments': line_segments,
        'junction_points': junction_points,
        'text_objects': text_objects,
        'net_names': net_names
    }

def check_connection(analysis_result, entity1, entity2):
    """
    Check if two entities are connected
    """
    if analysis_result is None:
        return False
    
    dsu = analysis_result['dsu']
    rep_name = analysis_result['rep_name']
    
    connected = are_connected(dsu, rep_name, entity1, entity2)
    print(f"Are '{entity1}' and '{entity2}' connected? {connected}")
    return connected

### ==============  Test code or visualization  ====================
# ######################  ##########################################
def visualize_connectivity(analysis_result, image_path, output_path=None):
    """
    Visualize the connectivity graph on the original image
    """
    if analysis_result is None:
        print("No analysis result to visualize")
        return
    
    import cv2
    import numpy as np
    import random
    
    # Load original image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Create visualization image
    vis_img = img.copy()
    
    dsu = analysis_result['dsu']
    rep_name = analysis_result['rep_name']
    line_segments = analysis_result['line_segments']
    junction_points = analysis_result['junction_points']
    text_objects = analysis_result['text_objects']
    
    # Generate colors for each connected component
    components = {}
    colors = {}
    
    # Find all unique components
    all_points = set()
    for sid, x1, y1, x2, y2 in line_segments:
        all_points.add(norm_point(x1, y1))
        all_points.add(norm_point(x2, y2))
    
    for point in all_points:
        root = dsu.find(point)
        if root not in components:
            components[root] = []
            # Generate random color for this component
            colors[root] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        components[root].append(point)
    
    print(f"Found {len(components)} connected components")
    
    # Draw line segments colored by component
    for sid, x1, y1, x2, y2 in line_segments:
        p1 = norm_point(x1, y1)
        root = dsu.find(p1)
        color = colors[root]
        cv2.line(vis_img, (x1, y1), (x2, y2), color, 3)
    
    # Draw junction points
    for jx, jy in junction_points:
        jp = norm_point(jx, jy)
        root = dsu.find(jp)
        color = colors.get(root, (255, 255, 255))
        cv2.circle(vis_img, (jx, jy), 8, color, -1)
        cv2.circle(vis_img, (jx, jy), 8, (0, 0, 0), 2)
    
    # Draw text labels with component colors
    for text_obj in text_objects:
        name = text_obj['text'].strip()
        if not name or len(name) < 2:
            continue
            
        x1, y1, x2, y2 = text_obj['bbox']
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        # Find which component this text belongs to
        text_color = (255, 255, 255)  # default white
        for root, net_name in rep_name.items():
            if net_name == name:
                text_color = colors.get(root, (255, 255, 255))
                break
        
        # Draw text background
        cv2.rectangle(vis_img, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), -1)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), text_color, 2)
        
        # Draw text
        cv2.putText(vis_img, name, (x1, y2-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # Add legend
    legend_y = 30
    for i, (root, net_name) in enumerate(rep_name.items()):
        color = colors.get(root, (255, 255, 255))
        cv2.rectangle(vis_img, (10, legend_y + i*25), (30, legend_y + i*25 + 15), color, -1)
        cv2.rectangle(vis_img, (10, legend_y + i*25), (30, legend_y + i*25 + 15), (0, 0, 0), 1)
        cv2.putText(vis_img, net_name, (35, legend_y + i*25 + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save visualization
    if output_path is None:
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_connectivity.png"
    
    cv2.imwrite(str(output_path), vis_img)
    print(f"Connectivity visualization saved to: {output_path}")
    
    return vis_img

def print_connectivity_report(analysis_result):
    """
    Print a detailed connectivity report
    """
    if analysis_result is None:
        print("No analysis result to report")
        return
    
    dsu = analysis_result['dsu']
    rep_name = analysis_result['rep_name']
    line_segments = analysis_result['line_segments']
    junction_points = analysis_result['junction_points']
    text_objects = analysis_result['text_objects']
    
    print("\n" + "="*50)
    print("CONNECTIVITY ANALYSIS REPORT")
    print("="*50)
    
    # Group components
    components = {}
    all_points = set()
    
    for sid, x1, y1, x2, y2 in line_segments:
        all_points.add(norm_point(x1, y1))
        all_points.add(norm_point(x2, y2))
    
    for point in all_points:
        root = dsu.find(point)
        if root not in components:
            components[root] = {'points': [], 'segments': [], 'net_names': []}
        components[root]['points'].append(point)
    
    # Add segments to components
    for sid, x1, y1, x2, y2 in line_segments:
        p1 = norm_point(x1, y1)
        root = dsu.find(p1)
        components[root]['segments'].append((sid, x1, y1, x2, y2))
    
    # Add net names to components
    for root, net_name in rep_name.items():
        if root in components:
            components[root]['net_names'].append(net_name)
    
    print(f"Total Connected Components: {len(components)}")
    print(f"Total Line Segments: {len(line_segments)}")
    print(f"Total Junction Points: {len(junction_points)}")
    print(f"Total Text Objects: {len(text_objects)}")
    print(f"Total Named Nets: {len(set(rep_name.values()))}")
    
    print("\nDETAILED COMPONENT BREAKDOWN:")
    print("-" * 30)
    
    for i, (root, comp_data) in enumerate(components.items(), 1):
        net_names = comp_data['net_names']
        num_segments = len(comp_data['segments'])
        num_points = len(comp_data['points'])
        
        print(f"\nComponent {i}:")
        print(f"  Root: {root}")
        print(f"  Net Names: {net_names if net_names else 'UNNAMED'}")
        print(f"  Points: {num_points}")
        print(f"  Line Segments: {num_segments}")
        
        if net_names:
            print(f"  This component connects: {' <-> '.join(net_names)}")
    
    # Connectivity matrix
    net_names = list(set(rep_name.values()))
    if len(net_names) > 1:
        print(f"\nCONNECTIVITY MATRIX:")
        print("-" * 20)
        print("Connected nets:")
        for i, name1 in enumerate(net_names):
            for j, name2 in enumerate(net_names[i+1:], i+1):
                connected = are_connected(dsu, rep_name, name1, name2)
                if connected:
                    print(f"  {name1} <-> {name2}")

if __name__ == "__main__":
    import glob
    # input_img = "./SIPI_example/cropped_objects/page_012/2/obj_2.png"
    input_images = glob.glob(f"pdf_pages_output/cropped_objects/*/*/obj_[0-9].png")

    for input_img in input_images:
        # Perform complete connectivity analysis
        result = analyze_connectivity(input_img)

        if result:
            # Generate detailed report
            print_connectivity_report(result)

            # Create visualization
            # visualize_connectivity(result, input_img)

            # Example connectivity checks
            # check_connection(result, "EN", "VRP3V3EC_VIN")
            # check_connection(result, "PVBATT", "VRP3V3EC_VIN")
            
            # # Check all possible pairs
            # net_names = result['net_names']
            # print(f"\nChecking all pairs from detected nets: {net_names}")
            # for i, name1 in enumerate(net_names):
            #     for j, name2 in enumerate(net_names[i+1:], i+1):
            #         check_connection(result, name1, name2)