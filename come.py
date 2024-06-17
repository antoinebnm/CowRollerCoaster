import glfw
import sys
import pdb
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import ArrayDatatype
import time
import numpy as np
import ctypes
from PIL.Image import open
import OBJ
from Ray import *
from scipy.spatial.transform import Rotation as R


# global variables
wld2cam=[]
cam2wld=[]
cow2wld=None
cursorOnCowBoundingBox=False
pickInfo=None
floorTexID=0
cameras= [
	[28,18,28, 0,2,0, 0,1,0],   
	[28,18,-28, 0,2,0, 0,1,0], 
	[-28,18,28, 0,2,0, 0,1,0], 
	[-12,12,0, 0,2,0, 0,1,0],  
	[0,100,0,  0,0,0, 1,0,0]
]
camModel=None
cowModel=None
H_DRAG=1
V_DRAG=2
# dragging state
isDrag=0
controlPoints = []
animStartTime = None
trackLength = 0
placedCows = []  # List to store transformation matrices of placed cows


class PickInfo:
    def __init__(self, cursorRayT, cowPickPosition, cowPickConfiguration, cowPickPositionLocal):
        self.cursorRayT=cursorRayT
        self.cowPickPosition=cowPickPosition.copy()
        self.cowPickConfiguration=cowPickConfiguration.copy()
        self.cowPickPositionLocal=cowPickPositionLocal.copy()

def vector3(x,y,z):
    return np.array((x,y,z))
def position3(v):
    # divide by w
    w=v[3]
    return vector3(v[0]/w, v[1]/w, v[2]/w)

def vector4(x,y,z):
    return np.array((x,y,z,1))

def rotate(m,v):
    return m[0:3, 0:3]@v
def transform(m, v):
    return position3(m@np.append(v,1))

def getTranslation(m):
    return m[0:3,3]
def setTranslation(m,v):
    m[0:3,3]=v

def makePlane( a,  b,  n):
    v=a.copy()
    for i in range(3):
        if n[i]==1.0:
            v[i]=b[i];
        elif n[i]==-1.0:
            v[i]=a[i];
        else:
            assert(n[i]==0.0);
            
    return Plane(rotate(cow2wld,n),transform(cow2wld,v));

def onKeyPress( window, key, scancode, action, mods):
    global cameraIndex
    if action==glfw.RELEASE:
        return ; # do nothing
    # If 'c' or space bar are pressed, alter the camera.
    # If a number is pressed, alter the camera corresponding the number.
    if key==glfw.KEY_C or key==glfw.KEY_SPACE:
        print( "Toggle camera %s\n"% cameraIndex );
        cameraIndex += 1;

    if cameraIndex >= len(wld2cam):
        cameraIndex = 0;

def drawOtherCamera():
    global cameraIndex,wld2cam, camModel
    for i in range(len(wld2cam)):
        if (i != cameraIndex):
            glPushMatrix();												# Push the current matrix on GL to stack. The matrix is wld2cam[cameraIndex].matrix().
            glMultMatrixd(cam2wld[i].T)
            drawFrame(5);											# Draw x, y, and z axis.
            frontColor = [0.2, 0.2, 0.2, 1.0];
            glEnable(GL_LIGHTING);									
            glMaterialfv(GL_FRONT, GL_AMBIENT, frontColor);			# Set ambient property frontColor.
            glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor);			# Set diffuse property frontColor.
            glScaled(0.5,0.5,0.5);										# Reduce camera size by 1/2.
            glTranslated(1.1,1.1,0.0);									# Translate it (1.1, 1.1, 0.0).
            camModel.render()
            glPopMatrix();												# Call the matrix on stack. wld2cam[cameraIndex].matrix() in here.

def drawFrame(leng):
    glDisable(GL_LIGHTING);	# Lighting is not needed for drawing axis.
    glBegin(GL_LINES);		# Start drawing lines.
    glColor3d(1,0,0);		# color of x-axis is red.
    glVertex3d(0,0,0);			
    glVertex3d(leng,0,0);	# Draw line(x-axis) from (0,0,0) to (len, 0, 0). 
    glColor3d(0,1,0);		# color of y-axis is green.
    glVertex3d(0,0,0);			
    glVertex3d(0,leng,0);	# Draw line(y-axis) from (0,0,0) to (0, len, 0).
    glColor3d(0,0,1);		# color of z-axis is  blue.
    glVertex3d(0,0,0);
    glVertex3d(0,0,leng);	# Draw line(z-axis) from (0,0,0) - (0, 0, len).
    glEnd();			# End drawing lines.

#*********************************************************************************
# Draw 'cow' object.
#*********************************************************************************/
def drawCow(_cow2wld, drawBB):

    glPushMatrix();		# Push the current matrix of GL into stack. This is because the matrix of GL will be change while drawing cow.

    # The information about location of cow to be drawn is stored in cow2wld matrix.
    # (Project2 hint) If you change the value of the cow2wld matrix or the current matrix, cow would rotate or move.
    glMultMatrixd(_cow2wld.T)

    drawFrame(5);										# Draw x, y, and z axis.
    frontColor = [0.25, 0.41, 0.88, 1.0];
    glEnable(GL_LIGHTING);
    glMaterialfv(GL_FRONT, GL_AMBIENT, frontColor);		# Set ambient property frontColor.
    glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor);		# Set diffuse property frontColor.
    cowModel.render()	# Draw cow. 
    glDisable(GL_LIGHTING);
    if drawBB:
        glBegin(GL_LINES);
        glColor3d(1,1,1);
        cow=cowModel
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmax[2]);

        glColor3d(1,1,1);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmax[2]);

        glColor3d(1,1,1);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmax[2]);


        glColor3d(1,1,1);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmax[2]);

        glColor3d(1,1,1);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmin[1], cow.bbmax[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmin[0], cow.bbmax[1], cow.bbmax[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmin[2]);
        glVertex3d( cow.bbmax[0], cow.bbmax[1], cow.bbmax[2]);
        glEnd();
    glPopMatrix();			# Pop the matrix in stack to GL. Change it the matrix before drawing cow.
def drawFloor():

    glDisable(GL_LIGHTING);

    # Set color of the floor.
    # Assign checker-patterned texture.
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, floorTexID );

    # Draw the floor. Match the texture's coordinates and the floor's coordinates resp. 
    nrep=4
    glBegin(GL_POLYGON);
    glTexCoord2d(0,0);
    glVertex3d(-12,-0.1,-12);		# Texture's (0,0) is bound to (-12,-0.1,-12).
    glTexCoord2d(nrep,0);
    glVertex3d( 12,-0.1,-12);		# Texture's (1,0) is bound to (12,-0.1,-12).
    glTexCoord2d(nrep,nrep);
    glVertex3d( 12,-0.1, 12);		# Texture's (1,1) is bound to (12,-0.1,12).
    glTexCoord2d(0,nrep);
    glVertex3d(-12,-0.1, 12);		# Texture's (0,1) is bound to (-12,-0.1,12).
    glEnd();

    glDisable(GL_TEXTURE_2D);	
    drawFrame(5);				# Draw x, y, and z axis.

def catmull_rom_spline(P0, P1, P2, P3, t):
    """
    Computes the point on a Catmull-Rom spline at parameter t.
    """
    return 0.5 * (
        (2 * P1) +
        (-P0 + P2) * t +
        (2 * P0 - 5 * P1 + 4 * P2 - P3) * t**2 +
        (-P0 + 3 * P1 - 3 * P2 + P3) * t**3
    )



import numpy as np

def display():
    global cameraIndex, cow2wld, animStartTime, controlPoints, trackLength, placedCows
    glClearColor(0.8, 0.9, 0.9, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the screen
    # set viewing transformation.
    glLoadMatrixd(wld2cam[cameraIndex].T)

    drawOtherCamera()  # Locate the camera's position, and draw all of them.
    drawFloor()  # Draw floor.

    if animStartTime is None:
        # Draw placed cows only if the animation has not started
        for cowMatrix in placedCows:
            drawCow(cowMatrix, False)
    else:
        animTime = glfw.get_time() - animStartTime
        speed_factor = 50.0  # Adjust this value to increase or decrease the speed
        laps = 3  # Number of laps to complete
        totalLength = trackLength * laps  # Total length representing three laps
        normalizedTime = (animTime * speed_factor) / totalLength  # Normalize time to fit within three laps

        if normalizedTime >= 3:
            # Reset state after three laps
            animStartTime = None
            controlPoints = []
            placedCows = []
            cow2wld = np.eye(4)
            glPushMatrix()
            glLoadIdentity()
            glTranslated(0, -cowModel.bbmin[1], -8)
            glRotated(-90, 0, 1, 0)
            cow2wld = glGetDoublev(GL_MODELVIEW_MATRIX).T
            glPopMatrix()
            return

        t = normalizedTime % 1  # Normalize t within the current lap
        currentLap = int(normalizedTime)  # Determine which lap we're on
        segment = int(t * len(controlPoints)) % len(controlPoints)
        u = (t * len(controlPoints)) % 1

        p0 = controlPoints[(segment - 1) % len(controlPoints)]
        p1 = controlPoints[segment]
        p2 = controlPoints[(segment + 1) % len(controlPoints)]
        p3 = controlPoints[(segment + 2) % len(controlPoints)]

        cowPos = catmull_rom_spline(p0, p1, p2, p3, u)
        print('pos', cowPos)

        # Compute the direction the cow is facing
        next_u = u + 0.01  # small increment to find the next position
        if next_u > 1:
            next_segment = (segment + 1) % len(controlPoints)
            next_u -= 1
        else:
            next_segment = segment

        next_p0 = controlPoints[(next_segment - 1) % len(controlPoints)]
        next_p1 = controlPoints[next_segment]
        next_p2 = controlPoints[(next_segment + 1) % len(controlPoints)]
        next_p3 = controlPoints[(next_segment + 2) % len(controlPoints)]

        next_cowPos = catmull_rom_spline(next_p0, next_p1, next_p2, next_p3, next_u)

        direction = next_cowPos - cowPos
        direction /= np.linalg.norm(direction)

        print('direction', direction)

        # Compute the right and up vectors
        up = np.array([0, 1, 0])
        right = np.cross(up, direction)
        print("up",up)
        print("right",right)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)

        rotation_matrix = np.eye(4)
        rotation_matrix[0, 0:3] = right
        rotation_matrix[1, 0:3] = up
        rotation_matrix[2, 0:3] = direction

        cow2wld = rotation_matrix
        print("cow2wld",cow2wld)
        print("cowPos",cowPos)
        setTranslation(cow2wld, cowPos)

    drawCow(cow2wld, cursorOnCowBoundingBox)  # Draw cow.

    glFlush()





def reshape(window, w, h):
    width = w;
    height = h;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);            # Select The Projection Matrix
    glLoadIdentity();                       # Reset The Projection Matrix
    # Define perspective projection frustum
    aspect = width/(float)(height);
    gluPerspective(45, aspect, 1, 1024);
    matProjection=glGetDoublev(GL_PROJECTION_MATRIX).T
    glMatrixMode(GL_MODELVIEW);             # Select The Modelview Matrix
    glLoadIdentity();                       # Reset The Projection Matrix

def initialize(window):
    global cursorOnCowBoundingBox, floorTexID, cameraIndex, camModel, cow2wld, cowModel, placedCows
    cursorOnCowBoundingBox = False
    # Set up OpenGL state
    glShadeModel(GL_SMOOTH)  # Set Smooth Shading
    glEnable(GL_DEPTH_TEST)  # Enables Depth Testing
    glDepthFunc(GL_LEQUAL)  # The Type Of Depth Test To Do
    # Use perspective correct interpolation if available
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    # Initialize the matrix stacks
    width, height = glfw.get_window_size(window)
    reshape(window, width, height)
    # Define lighting for the scene
    lightDirection = [1.0, 1.0, 1.0, 0]
    ambientIntensity = [0.1, 0.1, 0.1, 1.0]
    lightIntensity = [0.9, 0.9, 0.9, 1.0]
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientIntensity)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightIntensity)
    glLightfv(GL_LIGHT0, GL_POSITION, lightDirection)
    glEnable(GL_LIGHT0)

    # initialize floor
    im = open('bricks.bmp')
    try:
        ix, iy, image = im.size[0], im.size[1], im.tobytes("raw", "RGB", 0, -1)
    except SystemError:
        ix, iy, image = im.size[0], im.size[1], im.tobytes("raw", "RGBX", 0, -1)

    # Make texture which is accessible through floorTexID.
    floorTexID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, floorTexID)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, ix, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    # initialize cow
    cowModel = OBJ.OBJrenderer("cow.obj")

    # initialize cow2wld matrix
    glPushMatrix()  # Push the current matrix of GL into stack.
    glLoadIdentity()  # Set the GL matrix Identity matrix.
    glTranslated(0, -cowModel.bbmin[1], -8)  # Set the location of cow.
    glRotated(-90, 0, 1, 0)  # Set the direction of cow. These information are stored in the matrix of GL.
    cow2wld = glGetDoublev(GL_MODELVIEW_MATRIX).T  # convert column-major to row-major
    glPopMatrix()  # Pop the matrix on stack to GL.

    # intialize camera model.
    camModel = OBJ.OBJrenderer("camera.obj")

    # initialize camera frame transforms.
    cameraCount = len(cameras)
    for i in range(cameraCount):
        # 'c' points the coordinate of i-th camera.
        c = cameras[i]
        glPushMatrix()  # Push the current matrix of GL into stack.
        glLoadIdentity()  # Set the GL matrix Identity matrix.
        gluLookAt(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8])  # Setting the coordinate of camera.
        wld2cam.append(glGetDoublev(GL_MODELVIEW_MATRIX).T)
        glPopMatrix()  # Transfer the matrix that was pushed the stack to GL.
        cam2wld.append(np.linalg.inv(wld2cam[i]))
    cameraIndex = 0
    placedCows = []  # Ini

def calculate_track_length(control_points):
    length = 0
    for i in range(1, len(control_points)):
        length += np.linalg.norm(control_points[i] - control_points[i - 1])
    length += np.linalg.norm(control_points[-1] - control_points[0])  # Assuming it's a loop
    return length

def onMouseButton(window, button, state, mods):
    global isDrag, V_DRAG, H_DRAG, controlPoints, cow2wld, animStartTime, trackLength, placedCows
    GLFW_DOWN=1
    GLFW_UP=0
    x, y = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_LEFT:
        if state == GLFW_DOWN:
            if isDrag == H_DRAG:
                isDrag = 0
            else:
                isDrag = V_DRAG
            print("Left mouse down-click at %d %d\n" % (x, y))
            # start vertical dragging
        elif state == GLFW_UP and isDrag != 0:
            isDrag = H_DRAG
            print("Left mouse up\n")
            # start horizontal dragging using mouse-move events.
            if cursorOnCowBoundingBox and len(controlPoints) < 6:
                controlPoints.append(getTranslation(cow2wld))
                placedCows.append(cow2wld.copy())  # Store the current cow's transformation matrix
                print("Added control point:", getTranslation(cow2wld))
                if len(controlPoints) == 6:
                    trackLength = calculate_track_length(controlPoints)
                    animStartTime = glfw.get_time()
                    print("All control points added. Animation started.")
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        if state == GLFW_DOWN:
            print("Right mouse click at (%d, %d)\n" % (x, y))



# Variable globale pour enregistrer la position du curseur lorsque "Ctrl" est enfoncée
ctrlPickPosition = None

def onMouseDrag(window, x, y):
    global isDrag, cursorOnCowBoundingBox, pickInfo, cow2wld, ctrlPickPosition

    speed_factor = 2.0  # Adjust this value to increase or decrease the speed

    if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS:
        # Enregistrer la position du curseur lorsqu'on appuie sur "Ctrl"
        if ctrlPickPosition is None:
            ray = screenCoordToRay(window, x, y)
            pp = pickInfo
            draggingPlane = Plane(rotate(cow2wld, np.array((0, 0, 1))), pp.cowPickPosition)
            intersectResult = ray.intersectsPlane(draggingPlane)
            if intersectResult[0]:
                ctrlPickPosition = ray.getPoint(intersectResult[1])
    
    if isDrag: 
        print("in drag mode %d\n" % isDrag)
        if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS:
            # Déplacement vertical en utilisant la position enregistrée
            if cursorOnCowBoundingBox and ctrlPickPosition is not None:
                ray = screenCoordToRay(window, x, y)
                pp = pickInfo
                draggingPlane = Plane(rotate(cow2wld, np.array((0, 0, 1))), ctrlPickPosition)
                intersectResult = ray.intersectsPlane(draggingPlane)
                
                if intersectResult[0]:
                    currentPos = ray.getPoint(intersectResult[1])
                    verticalMove = (currentPos - pp.cowPickPosition) * speed_factor
                    verticalMove[0] = 0
                    verticalMove[2] = 0

                    T = np.eye(4)
                    setTranslation(T, verticalMove)
                    cow2wld = T @ pp.cowPickConfiguration
                print('vdrag with Ctrl')

        elif isDrag == V_DRAG:
            # Déplacement vertical sans Ctrl
            if cursorOnCowBoundingBox:
                ray = screenCoordToRay(window, x, y)
                pp = pickInfo
                draggingPlane = Plane(rotate(cow2wld, np.array((0, 0, 1))), pp.cowPickPosition)
                intersectResult = ray.intersectsPlane(draggingPlane)
                
                if intersectResult[0]:
                    currentPos = ray.getPoint(intersectResult[1])
                    verticalMove = (currentPos - pp.cowPickPosition) * speed_factor
                    verticalMove[0] = 0
                    verticalMove[2] = 0

                    T = np.eye(4)
                    setTranslation(T, verticalMove)
                    cow2wld = T @ pp.cowPickConfiguration
                print('vdrag')

        else:
            # Déplacement horizontal
            if cursorOnCowBoundingBox:
                ray = screenCoordToRay(window, x, y)
                pp = pickInfo
                p = Plane(np.array((0, 1, 0)), pp.cowPickPosition)
                c = ray.intersectsPlane(p)

                currentPos = ray.getPoint(c[1])
                print(pp.cowPickPosition, currentPos)
                print(pp.cowPickConfiguration, cow2wld)

                T = np.eye(4)
                horizontalMove = (currentPos - pp.cowPickPosition) * speed_factor
                setTranslation(T, horizontalMove)
                cow2wld = T @ pp.cowPickConfiguration
    else:
        ray = screenCoordToRay(window, x, y)

        planes = []
        cow = cowModel
        bbmin = cow.bbmin
        bbmax = cow.bbmax

        planes.append(makePlane(bbmin, bbmax, vector3(0, 1, 0)))
        planes.append(makePlane(bbmin, bbmax, vector3(0, -1, 0)))
        planes.append(makePlane(bbmin, bbmax, vector3(1, 0, 0)))
        planes.append(makePlane(bbmin, bbmax, vector3(-1, 0, 0)))
        planes.append(makePlane(bbmin, bbmax, vector3(0, 0, 1)))
        planes.append(makePlane(bbmin, bbmax, vector3(0, 0, -1)))

        o = ray.intersectsPlanes(planes)
        cursorOnCowBoundingBox = o[0]
        cowPickPosition = ray.getPoint(o[1])
        cowPickLocalPos = transform(np.linalg.inv(cow2wld), cowPickPosition)
        pickInfo = PickInfo(o[1], cowPickPosition, cow2wld, cowPickLocalPos)

        # Réinitialiser ctrlPickPosition lorsque la souris cesse de glisser
        ctrlPickPosition = None




def screenCoordToRay(window, x, y):
    width, height = glfw.get_window_size(window)

    matProjection=glGetDoublev(GL_PROJECTION_MATRIX).T
    matProjection=matProjection@wld2cam[cameraIndex]; # use @ for matrix mult.
    invMatProjection=np.linalg.inv(matProjection);
    # -1<=v.x<1 when 0<=x<width
    # -1<=v.y<1 when 0<=y<height
    vecAfterProjection =vector4(
            (float(x - 0))/(float(width))*2.0-1.0,
            -1*(((float(y - 0))/float(height))*2.0-1.0),
            -10)

    #std::cout<<"cowPosition in clip coordinate (NDC)"<<matProjection*cow2wld.getTranslation()<<std::endl;
	
    vecBeforeProjection=position3(invMatProjection@vecAfterProjection);

    rayOrigin=getTranslation(cam2wld[cameraIndex])
    return Ray(rayOrigin, normalize(vecBeforeProjection-rayOrigin))

def main():
    if not glfw.init():
        print ('GLFW initialization failed')
        sys.exit(-1)
    width = 800;
    height = 600;
    window = glfw.create_window(width, height, 'modern opengl example', None, None)
    if not window:
        glfw.terminate()
        sys.exit(-1)

    glfw.make_context_current(window)
    glfw.set_key_callback(window, onKeyPress)
    glfw.set_mouse_button_callback(window, onMouseButton)
    glfw.set_cursor_pos_callback(window, onMouseDrag)
    glfw.swap_interval(1)

    initialize(window);						
    while not glfw.window_should_close(window):
        glfw.poll_events()
        display()

        glfw.swap_buffers(window)

    glfw.terminate()
if __name__ == "__main__":
    main()
