To represent a 2D image as a 3D tensor model (often called a 3D displacement map or heightmap), you translate pixel coordinates $(X, Y)$ into spatial base grids and map the pixel brightness/intensity to the depth axis ($Z$).
To make this model move, rotate, and scale, you use a sequence of core matrix operations. Here is the mathematical framework followed by an executable Python code example.
------------------------------
## The 3D Tensor Transformations Pipeline## 1. Matrix Addition (Translation / Centering)
To spin your image mesh around its true geometric center rather than its top-left corner, you must translate your raw $(X, Y, Z)$ coordinates. Matrix addition shifts every coordinate point uniformly.
$$\begin{bmatrix} X_{new} \\ Y_{new} \\ Z_{new} \end{bmatrix} = \begin{bmatrix} X_{orig} \\ Y_{orig} \\ Z_{orig} \end{bmatrix} + \begin{bmatrix} T_x \\ T_y \\ T_z \end{bmatrix}$$ 

* Example: If your image is $200 \times 200$ pixels, you add $\begin{bmatrix} -100 & -100 & 0 \end{bmatrix}^T$ to center it at the $(0,0)$ origin.

## 2. Matrix Multiplication (Scaling)
To control the physical size of your 3D image or amplify its depth (making features pop out further along the Z-axis), you multiply the coordinates by a scaling matrix.
$$\begin{bmatrix} X_{new} \\ Y_{new} \\ Z_{new} \end{bmatrix} = \begin{bmatrix} S_x & 0 & 0 \\ 0 & S_y & 0 \\ 0 & 0 & S_z \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$ 
## 3. Matrix Multiplication (Rotation Transformations)
To rotate your 3D image mesh through space, you multiply your vertex positions by trigonometric rotation matrix operators.

* Rotation around X-axis (Pitch / Tilting up and down):
$$R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}$$ 
* Rotation around Y-axis (Yaw / Turning side to side):
$$R_y(\phi) = \begin{bmatrix} \cos\phi & 0 & \sin\phi \\ 0 & 1 & 0 \\ -\sin\phi & 0 & \cos\phi \end{bmatrix}$$ 

------------------------------
## Python Implementation Example (Using NumPy)
This self-contained script generates a synthetic image (a bright circle on a dark background), converts it into a 3D structural heightmap tensor, centers it using matrix addition, and rotates it using matrix multiplication.

import numpy as npimport matplotlib.pyplot as pltfrom mpl_toolkits.mplot3d import Axes3D
# =====================================================================# 1. GENERATE SYNTHETIC IMAGE DATA (2D TENSOR)# =====================================================================resolution = 50x_indices = np.linspace(-1, 1, resolution)y_indices = np.linspace(-1, 1, resolution)X_grid, Y_grid = np.meshgrid(x_indices, y_indices)
# Create a smooth, bright circular protrusion in the center (intensity = Z height)distance_from_center = np.sqrt(X_grid**2 + Y_grid**2)Z_intensity = np.exp(-(distance_from_center**2) / 0.15) 
# Flatten grids into a clean multi-dimensional structural vertex tensor: [N, 3]vertices = np.vstack([X_grid.flatten(), Y_grid.flatten(), Z_intensity.flatten()]).T
# =====================================================================# 2. MATRIX ADDITION (TRANSLATION)# =====================================================================# Let's shift our 3D image slightly higher up on the Y axis and deeper back on Ztranslation_vector = np.array([0.0, 0.5, -0.2])translated_vertices = vertices + translation_vector  # Broadcasted Matrix Addition
# =====================================================================# 3. MATRIX MULTIPLICATION (ROTATION TRANSFORMATIONS)# =====================================================================def get_3d_rotation_matrix(angle_x_deg, angle_y_deg):
    rad_x = np.radians(angle_x_deg)
    rad_y = np.radians(angle_y_deg)
    
    # Rotation around X operator
    Rx = np.array([,
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x),  np.cos(rad_x)]
    ])
    
    # Rotation around Y operator
    Ry = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
,
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ])
    
    # Combined transformation matrix via dot product multiplication
    return np.dot(Rx, Ry)
# Get rotation configuration (Tilt 35 degrees down, rotate 45 degrees sideways)Rotation_Matrix = get_3d_rotation_matrix(angle_x_deg=35, angle_y_deg=45)
# Transform the vertices tensor using Matrix Multiplication# We multiply by the transpose (.T) because our vertex array shape layout is [N, 3]rotated_vertices = np.dot(translated_vertices, Rotation_Matrix.T)
# =====================================================================# 4. PLOT VISUALIZATION# =====================================================================# Reshape flattened array maps back to 2D coordinates for rendering meshesX_final = rotated_vertices[:, 0].reshape(resolution, resolution)Y_final = rotated_vertices[:, 1].reshape(resolution, resolution)Z_final = rotated_vertices[:, 2].reshape(resolution, resolution)
fig = plt.figure(figsize=(10, 5))
# Plot 1: Raw 2D Input Image Intensityax1 = fig.add_subplot(121)
ax1.imshow(Z_intensity, cmap='plasma')
ax1.set_title("Original 2D Image Tensor")
ax1.axis('off')
# Plot 2: Transformed 3D Heightmap Meshax2 = fig.add_subplot(122, projection='3d')# Surface colors shift dynamically mapping directly to structural Z vector depth
ax2.plot_surface(X_final, Y_final, Z_final, cmap='plasma', edgecolor='none', alpha=0.9)
ax2.set_title("Transformed 3D Image Heightmap")
ax2.set_zlim([-1, 1])

plt.tight_layout()
plt.show()

## Integration with your Flask Framework
If you port this code into your existing Flask + Canvas stack:

   1. Your Python code will open a real image using PIL.Image or cv2.
   2. Convert it to grayscale to read brightness channels as a 2D pixel array matrix.
   3. Perform the exact matrix calculations shown above on the backend.
   4. Send the X_final and Y_final coordinate properties out as a simple JSON coordinate package to your HTML5 Canvas layout.

Would you like to extend this to load a real JPEG or PNG image file from your local computer into the Python heightmap transformation code, or should we implement surface lighting shadows based on these mathematical vectors?

import math
import io
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

# Global runtime cache to keep active mesh tensors in memory
MAP_VERTICES = []
MAP_FACES = []

def process_image_to_heightmap(image_bytes, target_resolution=40):
    """Parses raw image files, transforms them to grayscale arrays, and builds a 3D tensor grid"""
    global MAP_VERTICES, MAP_FACES
    
    # 1. Load image and convert it into a normalized grayscale luminosity map
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Downscale the structural sampling frequency grid to prevent canvas rendering lags
    img = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
    pixels = img.load()
    
    new_vertices = []
    new_faces = []
    
    # 2. Build 3D coordinate space displacement landscape
    for i in range(target_resolution):
        y_percent = i / (target_resolution - 1)
        y_pos = (y_percent * 2.0) - 1.0  # Normalized spacing bounds (-1.0 to 1.0)
        
        for j in range(target_resolution):
            x_percent = j / (target_resolution - 1)
            x_pos = (x_percent * 2.0) - 1.0
            
            # Read brightness intensity scale (0-255 mapped down to 0.0-1.0)
            brightness = pixels[j, i] / 255.0
            
            # Map structural luminosity straight to Z height axis depth displacement
            z_pos = brightness * 0.6  
            
            new_vertices.append([x_pos, y_pos, z_pos])
            
    # 3. Create matrix mesh connecting lines indices
    for i in range(target_resolution - 1):
        for j in range(target_resolution - 1):
            p0 = i * target_resolution + j
            p1 = i * target_resolution + (j + 1)
            p2 = (i + 1) * target_resolution + (j + 1)
            p3 = (i + 1) * target_resolution + j
            new_faces.append([p0, p1, p2, p3])
            
    MAP_VERTICES = new_vertices
    MAP_FACES = new_faces

def get_lighted_3d_mesh(angle_x, angle_y, light_x, light_y, light_z):
    # Fallback to a synthetic default pattern if no local files have been uploaded yet
    if not MAP_VERTICES:
        generate_fallback_mesh()

    rad_x = math.radians(angle_x)
    rad_y = math.radians(angle_y)

    cx, sx = math.cos(rad_x), math.sin(rad_x)
    cy, sy = math.cos(rad_y), math.sin(rad_y)

    # Matrix Multiplication: Apply dynamic rotations to geometry matrix positions
    rotated_vertices = []
    for x, y, z in MAP_VERTICES:
        # Rotation around Y axis
        x1 = x * cy + z * sy
        y1 = y
        z1 = -x * sy + z * cy
        
        # Rotation around X axis
        x2 = x1
        y2 = y1 * cx - z1 * sx
        z2 = y1 * sx + z1 * cx
        
        rotated_vertices.append([x2, y2, z2])

    # Standardize lighting position direction vector properties
    light_len = math.sqrt(light_x**2 + light_y**2 + light_z**2) or 1
    lx, ly, lz = light_x / light_len, light_y / light_len, light_z / light_len

    rendered_faces = []

    for face in MAP_FACES:
        p0 = rotated_vertices[face[0]]
        p1 = rotated_vertices[face[1]]
        p2 = rotated_vertices[face[2]]
        p3 = rotated_vertices[face[3]]

        # Normal vector tracking via matrix cross product
        v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
        v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]]
        
        normal = [
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]
        ]
        
        normal_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2) or 1
        nx, ny, nz = normal[0]/normal_len, normal[1]/normal_len, normal[2]/normal_len

        # Matrix Dot Product determines surface lighting brightness intensities
        dot_product = nx*lx + ny*ly + nz*lz
        
        ambient_light = 0.15
        diffuse_intensity = max(0.0, dot_product)
        total_illumination = ambient_light + (1.0 - ambient_light) * diffuse_intensity

        # Volumetric Bronze Palette shading treatments
        # r = int(220 * total_illumination)
        # g = int(140 * total_illumination)
        # b = int(60 * total_illumination)

        # Base Color: Hex #0ea5e9 / RGB (14, 165, 233)
        # r = int(14 * total_illumination)
        # g = int(165 * total_illumination)
        # b = int(233 * total_illumination)

        # Base Color: Hex #cbd5e1 / RGB (203, 213, 225)
        # r = int(203 * total_illumination)
        # g = int(213 * total_illumination)
        # b = int(225 * total_illumination)

        # Base Color: Hex #10b981 / RGB (16, 185, 129)
        # r = int(16 * total_illumination)
        # g = int(185 * total_illumination)
        # b = int(129 * total_illumination)

        # Base Color: Hex #f43f5e / RGB (244, 63, 94)
        r = int(244 * total_illumination)
        g = int(63 * total_illumination)
        b = int(94 * total_illumination)




        avg_z = (p0[2] + p1[2] + p2[2] + p3[2]) / 4.0

        face_points_2d = []
        for vert_idx in face:
            pt = rotated_vertices[vert_idx]
            scale = 130
            screen_x = pt[0] * scale + 200
            screen_y = -pt[1] * scale + 200 # Reverse canvas layout Y space
            face_points_2d.append({"x": screen_x, "y": screen_y})

        rendered_faces.append({
            "points": face_points_2d,
            "color": f"rgb({r}, {g}, {b})",
            "avg_z": avg_z
        })

    # Painter's Algorithm sorting structure prevents background face leaks
    rendered_faces.sort(key=lambda f: f["avg_z"])

    return {
        "faces": rendered_faces,
        "raw_tensor_state": {
            "angles": {"x": angle_x, "y": angle_y},
            "light": {"x": light_x, "y": light_y, "z": light_z}
        }
    }

def generate_fallback_mesh():
    """Generates a default geometric wave structure if no custom file has been uploaded"""
    res = 30
    fallback_bytes = io.BytesIO()
    img = Image.new('L', (res, res))
    pixels = img.load()
    for i in range(res):
        for j in range(res):
            dist = math.sqrt(((i/res)-0.5)**2 + ((j/res)-0.5)**2)
            pixels[j, i] = int((math.sin(dist * 12) + 1.0) * 127)
    img.save(fallback_bytes, format='PNG')
    process_image_to_heightmap(fallback_bytes.getvalue(), target_resolution=30)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/upload_image', methods=['POST'])
def upload_image_api():
    """Accepts local image files and passes them into our tensor extraction engine"""
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        # Parse pixel arrays into structured grid matrices
        process_image_to_heightmap(image_bytes, target_resolution=35)
        return jsonify({"success": True, "message": "Image tensor loaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/render', methods=['GET'])
def render_api():
    ax = float(request.args.get('x', 30))
    ay = float(request.args.get('y', 45))
    lx = float(request.args.get('lx', 1.0))
    ly = float(request.args.get('ly', 1.0))
    lz = float(request.args.get('lz', 1.0))
    
    return jsonify(get_lighted_3d_mesh(ax, ay, lx, ly, lz))

if __name__ == '__main__':
    app.run(debug=True, port=5000)

