import math

def distance_squared(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2

def is_lexically_smaller(c1, c2):
    # TA's Lexicographical tie-breaker rule
    if c1[0] != c2[0]: return c1[0] < c2[0]
    if c1[1] != c2[1]: return c1[1] < c2[1]
    return c1[2] < c2[2]

def c_div(a, b):
    # Pure integer division that truncates toward zero (exactly like C++)
    if a * b >= 0:
        return a // b
    else:
        return -(abs(a) // abs(b))

def verify_kmeans(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    n = int(lines[0].strip())
    k = int(lines[1].strip())
    T = int(lines[2].strip())
    
    # Parse points
    points = []
    for i in range(3, 3 + n):
        parts = list(map(int, lines[i].strip().split()))
        points.append({'x': parts[0], 'y': parts[1], 'z': parts[2], 'I': parts[3], 'id': i-3})
        
    # 1. Initialize centroids with the first k points
    centroids = [{'x': points[i]['x'], 'y': points[i]['y'], 'z': points[i]['z']} for i in range(k)]
    cluster_assignments = [-1] * n
    
    # 2. K-Means Iterations
    for iteration in range(T):
        changed = False
        
        # Assignment Step
        for p_idx, p in enumerate(points):
            min_dist = float('inf')
            best_c_idx = -1
            
            for c_idx, c in enumerate(centroids):
                dist = distance_squared((p['x'], p['y'], p['z']), (c['x'], c['y'], c['z']))
                
                if dist < min_dist:
                    min_dist = dist
                    best_c_idx = c_idx
                elif dist == min_dist:
                    # Tie-breaker: Lexically smaller centroid!
                    c_best = centroids[best_c_idx]
                    if is_lexically_smaller((c['x'], c['y'], c['z']), (c_best['x'], c_best['y'], c_best['z'])):
                        best_c_idx = c_idx
                        
            if cluster_assignments[p_idx] != best_c_idx:
                cluster_assignments[p_idx] = best_c_idx
                changed = True
                
        # Early exit if no points moved
        if not changed:
            break
            
        # Update Step (Integer Division)
        new_centroids = [{'x': 0, 'y': 0, 'z': 0, 'count': 0} for _ in range(k)]
        for p_idx, p in enumerate(points):
            c_idx = cluster_assignments[p_idx]
            new_centroids[c_idx]['x'] += points[p_idx]['x']
            new_centroids[c_idx]['y'] += points[p_idx]['y']
            new_centroids[c_idx]['z'] += points[p_idx]['z']
            new_centroids[c_idx]['count'] += 1
            
        for c_idx in range(k):
            count = new_centroids[c_idx]['count']
            if count > 0:
                # Standard Python integer division (//)
               centroids[c_idx]['x'] = c_div(new_centroids[c_idx]['x'], count)
               centroids[c_idx]['y'] = c_div(new_centroids[c_idx]['y'], count)
               centroids[c_idx]['z'] = c_div(new_centroids[c_idx]['z'], count)

    # 3. Histogram Equalization per Cluster
    output_lines = []
    for p_idx, p in enumerate(points):
        c_idx = cluster_assignments[p_idx]
        
        # Get intensities of all points in the same cluster
        cluster_intensities = [points[i]['I'] for i in range(n) if cluster_assignments[i] == c_idx]
        
        m = len(cluster_intensities)
        my_I = p['I']
        
        if m > 0:
            min_intensity = min(cluster_intensities)
            c_min = cluster_intensities.count(min_intensity)
            cdf_val = sum(1 for intensity in cluster_intensities if intensity <= my_I)
            
            if m == c_min:
                final_I = my_I
            else:
                mapped = ((cdf_val - c_min) / (m - c_min)) * 255.0
                final_I = min(255, max(0, math.floor(mapped)))
        else:
            final_I = my_I
            
        output_lines.append(f"{p['x']} {p['y']} {p['z']} {int(final_I)}")
        
    with open("kmeans_truth.txt", "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print("Generated python_kmeans_ground_truth.txt")

if __name__ == "__main__":
    verify_kmeans("input.txt")