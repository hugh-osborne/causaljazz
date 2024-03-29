import numpy as np
import os

import OpenGL.GL
import OpenGL.GLUT
import OpenGL.GLU

from PIL import Image
from PIL import ImageOps

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

class Visualiser:
    def __init__(self, nd=3, record_png=True, out_data_filename="out_data"):
        if nd != 3 and nd != 2:
            print("Visualiser can only render 2 or 3 dimensions. (", nd, " requested)")

        self.num_dimensions = nd
        self.closed = False
        
        self.record_png = record_png
        if self.record_png:
            self.out_data_filename = out_data_filename
            try:
                os.mkdir(self.out_data_filename)
            except OSError as error:
                print(error)
                print("Continuing...")
        
        self.file_iteration_num = 0
        
        self.rot_y = 0
        self.rot_x = 0
        self.zoom = -2
        
        self.eye_y_angle = 0.0
        self.eye_x_angle = 0.0
        self.eye_dist = 5.0
        
        self.left_held = False
        self.right_held = False
        self.up_held = False
        self.down_held = False
        self.pgup_held = False
        self.pgdn_held = False

        self.verts = (
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)
            )

        self.surfaces = tuple([(0,1,2,3)]
            )

        if self.num_dimensions == 3:
            self.verts = (
             (1, -1, -1),
             (1, 1, -1),
             (-1, 1, -1),
             (-1, -1, -1),
             (1, -1, 1),
             (1, 1, 1),
             (-1, -1, 1),
             (-1, 1, 1)
             )

            self.surfaces = (
                (0,1,2,3),
                (3,2,7,6),
                (6,7,5,4),
                (4,5,1,0),
                (1,5,7,2),
                (4,0,3,6)
                )

    def square(self, pos=(0,0), scale=(1.0,1.0), col=(1,0,0)):
        colours = (col,col,col,col)

        glBegin(GL_QUADS)
        for surface in self.surfaces:
          x = 0
          for vertex in surface:
            glColor3fv(colours[x])
            x += 1
            vert = [0 for v in self.verts[vertex]]
            vert[0] = (self.verts[vertex][0] * (scale[0]/2.0)) + pos[0]
            vert[1] = (self.verts[vertex][1] * (scale[1]/2.0)) + pos[1]
            if self.num_dimensions == 3:
                vert[2] = 0.0
                glVertex3fv(vert)
            else:
                glVertex2fv(vert)

        glEnd()

    def cube(self, pos=(0,0,0),scale=(1.0,1.0,1.0),col=(1,0,0,0.1)):
        colours = (col,col,col,col,col,col,col,col)

        glBegin(GL_QUADS)
        for surface in self.surfaces:
          x = 0
          for vertex in surface:
            glColor4fv(colours[x])
            x += 1
            vert = [0 for v in self.verts[vertex]]
            vert[0] = (self.verts[vertex][0] * (scale[0]/2.0)) + pos[0]
            vert[1] = (self.verts[vertex][1] * (scale[1]/2.0)) + pos[1]
            vert[2] = (self.verts[vertex][2] * (scale[2]/2.0)) + pos[2]
            glVertex3fv(vert)

        glEnd()
        
    def drawOutline(self):
        if self.num_dimensions != 3:
            return
        col =(0.5,0.5,0.5,0.1)
        colours = (col,col,col,col,col,col,col,col)
        
        glBegin(GL_QUADS)
        for surface in self.surfaces:
          x = 0
          for vertex in surface:
            glColor4fv(colours[x])
            x += 1
            vert = [0 for v in self.verts[vertex]]
            vert[0] = (self.verts[vertex][0] * 2.0)
            vert[1] = (self.verts[vertex][1] * 2.0)
            vert[2] = (self.verts[vertex][2] * 2.0)
            glVertex3fv(vert)

        glEnd()

    def drawCell3D(self, cell_coords, cell_mass, origin_location=(0.0,0.0,0.0), max_size=(2.0,2.0,2.0), max_res=(10,10,10)):
        if self.closed:
            return

        widths = (max_size[0]/max_res[0], max_size[1]/max_res[1], max_size[2]/max_res[2])
        pos = (origin_location[0] - (max_size[0]/2.0) + ((cell_coords[0]+0.5)*widths[0]), origin_location[1] - (max_size[1]/2.0) + ((cell_coords[1]+0.5)*widths[1]), origin_location[2] - (max_size[2]/2.0) + ((cell_coords[2]+0.5)*widths[2]))
        
        self.cube(pos, scale=widths, col=(1.0,0,0,cell_mass))

    def drawCell2D(self, cell_coords, cell_mass, origin_location=(0.0,0.0), max_size=(2.0,2.0), max_res=(10,10)):
        if self.closed:
            return

        widths = (max_size[0]/max_res[0], max_size[1]/max_res[1])
        pos = (origin_location[0] - (max_size[0]/2.0) + ((cell_coords[0]+0.5)*widths[0]), origin_location[1] - (max_size[1]/2.0) + ((cell_coords[1]+0.5)*widths[1]))

        inferno_values = [(1.0 / 8.0) * i for i in range(8)]        
        inferno_r_range = [0.0, 40/255, 101/255, 159/25, 212/255, 245/255, 250/255, 252/255]
        inferno_g_range = [0.0, 11/255, 21/255, 42/255, 72/255, 125/255, 193/255, 255/255]
        inferno_b_range = [4/255, 84/255, 110/255, 99/255, 66/255, 21/255, 39/255, 164/255]

        lower_num = int(cell_mass * 8.0)
        if lower_num > 6:
            lower_num = 6
            
        green = inferno_g_range[lower_num] + (((cell_mass - inferno_values[lower_num]) / (inferno_values[lower_num+1] - inferno_values[lower_num])) * (inferno_g_range[lower_num+1] - inferno_g_range[lower_num]))
        red = inferno_r_range[lower_num] + (((cell_mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_r_range[lower_num + 1] - inferno_r_range[lower_num]))
        blue = inferno_b_range[lower_num] + (((cell_mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_b_range[lower_num + 1] - inferno_b_range[lower_num]))
        
        self.square(pos, scale=widths, col=(red,green,blue))

    def drawCell1D(self, cell_coords, cell_mass, origin_location=(0.0), max_size=(2.0), max_res=(100)):
        if self.closed:
            return

        widths = (max_size[0]/max_res[0], 0.5)
        pos = (origin_location[0] - (max_size[0]/2.0) + ((cell_coords[0]+0.5)*widths[0]), 0.0)
        
        self.square(pos, scale=widths, col=(cell_mass,0,0))

    def drawCell(self, cell_coords, cell_mass, origin_location=(0.0), max_size=(2.0), max_res=(10)):
        if len(max_res) == 1:
            self.drawCell1D(cell_coords, cell_mass, origin_location, max_size, max_res)
        elif len(max_res) == 2:
            self.drawCell2D(cell_coords, cell_mass, origin_location, max_size, max_res)
        elif len(max_res) == 3:
            self.drawCell3D(cell_coords, cell_mass, origin_location, max_size, max_res)
        else:
            print("Trying to render too many dimensions.")


    def setupVisuliser(self, display_size=(800,800)):
        pygame.init()
        pygame.display.set_caption('Visualiser')
        display = display_size
        pygame.display.set_mode(display,DOUBLEBUF|OPENGL)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.num_dimensions == 3:
            gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
            #gluLookAt(0.0, 0.0, 2.0, 0, 0, 0, 0, 1, 0)
        else:
            glOrtho(-1.0, 1.0, -1.0, 1.0, 0.1, 50.0)
        
        glEnable(GL_BLEND)
        glTranslatef(0.0,0.0,-2)
        glRotatef(0, 0, 0, 0)

    def beginRendering(self):
        if self.closed:
            return False
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.closed = True
            if event.type == pygame.KEYDOWN and self.num_dimensions == 3:
                if event.key == pygame.K_LEFT:
                    self.left_held = True
                if event.key == pygame.K_RIGHT:
                    self.right_held = True
                if event.key == pygame.K_UP:
                    self.up_held = True
                if event.key == pygame.K_DOWN:
                    self.down_held = True
                if event.key == pygame.K_PAGEDOWN:
                    self.pgdn_held = True
                if event.key == pygame.K_PAGEUP:
                    self.pgup_held = True
            if event.type == pygame.KEYUP and self.num_dimensions == 3:
                if event.key == pygame.K_LEFT:
                    self.left_held = False
                if event.key == pygame.K_RIGHT:
                    self.right_held = False
                if event.key == pygame.K_UP:
                    self.up_held = False
                if event.key == pygame.K_DOWN:
                    self.down_held = False
                if event.key == pygame.K_PAGEDOWN:
                    self.pgdn_held = False
                if event.key == pygame.K_PAGEUP:
                    self.pgup_held = False
 
        rot_speed = 0.1
        zoom_speed = 0.1
        if self.pgup_held:
            self.eye_dist -= zoom_speed
        if self.pgdn_held:
            self.eye_dist += zoom_speed
        if self.left_held:
            self.eye_y_angle -= rot_speed
        if self.right_held:
            self.eye_y_angle += rot_speed
        if self.up_held:
            self.eye_x_angle += rot_speed
        if self.down_held:
            self.eye_x_angle -= rot_speed
            
        self.eye_y_angle = self.eye_y_angle % (2*np.pi)
        #self.eye_x_angle = self.eye_x_angle % (2*np.pi)

        if self.eye_x_angle > np.pi/2.0:
            self.eye_x_angle = np.pi/2.0
            
        if self.eye_x_angle < -np.pi/2.0:
            self.eye_x_angle = -np.pi/2.0
            
        new_eye_x = self.eye_dist * np.sin(self.eye_y_angle)
        new_eye_y = self.eye_dist * np.sin(self.eye_x_angle)
        new_eye_z = self.eye_dist * np.cos(self.eye_y_angle)
        
        if self.num_dimensions == 3:
            gluLookAt(new_eye_x, new_eye_y, new_eye_z, 0, 0, 0, 0, 1, 0)
        
        if self.closed:
            return False
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        return True


    def endRendering(self, iteration_num=None):
        if self.closed:
            return
        
        if iteration_num is not None:
            self.file_iteration_num = iteration_num
        
        self.drawOutline()
        
        if self.record_png:
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            data = glReadPixels(0, 0, 800,800, GL_RGBA, GL_UNSIGNED_BYTE)
            image = Image.frombytes("RGBA", (800,800), data)
            image = ImageOps.flip(image)
            image.save( self.out_data_filename + "/" + str(self.file_iteration_num) + '.png', 'PNG')

        pygame.display.flip()
        pygame.time.wait(10)
        
        self.file_iteration_num += 1

    def convertImageOutputToVideo(self, filename="movie", speed=30):
        os.system("ffmpeg -r " + str(speed) + " -i " + self.out_data_filename + "\\" + "%d.png -vcodec mpeg4 -q:v 0 -y " + filename + ".mp4")