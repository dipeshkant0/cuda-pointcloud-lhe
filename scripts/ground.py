import math

def distance_squared(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2

def verify_exact_knn(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    n = int(lines[0].strip())
    k = int(lines[1].strip())
    
    # Parse points: (x, y, z, I, original_index)
    points = []
    for i in range(3, 3 + n):
        parts = list(map(int, lines[i].strip().split()))
        points.append((parts[0], parts[1], parts[2], parts[3], i-3))
        
    output_lines = []
    
    for i, p_i in enumerate(points):
        neighbors = []
        for j, p_j in enumerate(points):
            if i == j:
                continue
            dist2 = distance_squared(p_i, p_j)
            # Tuple for sorting: (distance, x, y, z, original_index)
            # This perfectly matches the TA's exact tie-breaking rules!
            neighbors.append((dist2, p_j[0], p_j[1], p_j[2], p_j[4], p_j[3]))
            
        # Sort to find top K
        neighbors.sort()
        top_k = neighbors[:k]
        
        # 1. K+1 Histogram Rule (Include center point)
        neighborhood_intensities = [p_i[3]] + [n[5] for n in top_k]
        
        my_I = p_i[3]
        min_intensity = min(neighborhood_intensities)
        
        actual_k = len(neighborhood_intensities) # Should be k+1
        cdf_val = sum(1 for intensity in neighborhood_intensities if intensity <= my_I)
        c_min = sum(1 for intensity in neighborhood_intensities if intensity == min_intensity)
        
        # 2. Equation Application
        if actual_k == c_min:
            final_I = my_I
        else:
            mapped = ((cdf_val - c_min) / (actual_k - c_min)) * 255.0
            final_I = min(255, max(0, math.floor(mapped)))
            
        output_lines.append(f"{p_i[0]} {p_i[1]} {p_i[2]} {int(final_I)}")
        
    with open("python_ground_truth.txt", "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print("Generated python_ground_truth.txt")

if __name__ == "__main__":
    verify_exact_knn("input.txt")