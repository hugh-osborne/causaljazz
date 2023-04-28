#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>
#include <thread>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <boost/filesystem.hpp>
#include <iostream>

#include "display.hpp"
#include "glm/glm.hpp"

#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT 500
#define SHOW_TEXT true

using namespace std::chrono;

Display* Display::disp = 0;

Display::Display(){
	lastTime = 0;
	write_frames = false;
	start_time = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
	_dws = std::map<unsigned int, DisplayWindow>();
	upPressed = false;
	downPressed = false;
	leftPressed = false;
	rightPressed = false;
	pgdnPressed = false;
	pgupPressed = false;
}

Display::~Display(){
	if (glutGetWindow()){
		Display::getInstance()->updateDisplay(1);
		glutDestroyWindow(glutGetWindow());
	}
}

unsigned int Display::addPopulation(unsigned int nid, PopulationHelper* pop) {
	unsigned int index = _dws.size();

	DisplayWindow window;
	window.population = pop;
	window.mass_population = 0;
	window.dim0 = 0;
	window.dim1 = 1;
	window.max_mass = -9999999;
	window.min_mass = 9999999;

	std::map<unsigned int,DisplayWindow>::iterator it = _dws.find(nid);
	if(it == _dws.end())
		_dws.insert(std::make_pair(nid, window));

	return index;
}

unsigned int Display::addMassPopulation(unsigned int nid, MassPopulation* pop) {
	unsigned int index = _dws.size();

	DisplayWindow window;
	window.mass_population = pop;
	window.population = 0;
	window.dim0 = 0;
	window.dim1 = 1;
	window.max_mass = -9999999;
	window.min_mass = 9999999;

	std::map<unsigned int, DisplayWindow>::iterator it = _dws.find(nid);
	if (it == _dws.end())
		_dws.insert(std::make_pair(nid, window));

	return index;
}

void Display::calculateMarginal(double& total, std::vector<unsigned int> remaining_dims,
	std::vector<unsigned int> coords, PopulationHelper* pop, MassPopulation* mass_pop,
	std::vector<unsigned int>& cells, NdGrid* grid) {

	if (remaining_dims.size() == 0) {
		if (pop)
			total += (double)cells[pop->getGridOnCard()->getGrid()->getCellNum(coords)] / (double)pop->getNumNeurons();
		if (mass_pop)
			total += ((double)mass_pop->getMass()[grid->getCellNum(coords)]);
		return;
	}

	for (unsigned int c = 0; c < grid->getRes()[remaining_dims.back()]; c++) {
		coords[remaining_dims.back()] = c;
		if (remaining_dims.size() <= 1) {
			unsigned int idx = grid->getCellNum(coords);
			double mass = 0.0;

			if (pop)
				mass = 1.0 - ((double)cells[pop->getGridOnCard()->getGrid()->getCellNum(coords)] / (double)pop->getNumNeurons());

			if (mass_pop)
				mass = ((double)mass_pop->getMass()[idx]);

			total += mass;
		}
		else {
			remaining_dims.pop_back();
			calculateMarginal(total, remaining_dims, coords, pop, mass_pop, cells, grid);
		}
	}
}

// The OpenGL display function, called as fast as possible (ish).
void Display::display(void) {
	if (_dws.size() == 0)
		return;
	milliseconds real_time = duration_cast< milliseconds >(
    system_clock::now().time_since_epoch());
	milliseconds time_elapsed = real_time - start_time;

	int window_index = 0;
	for (std::map<unsigned int, DisplayWindow>::iterator iter = Display::getInstance()->_dws.begin(); iter != Display::getInstance()->_dws.end(); ++iter){
		if (iter->second._window_index == glutGetWindow())
			window_index = iter->first;
	}

	// if (time_elapsed.count() % 10 != 0)
	// 	return;

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_QUADS);
	
	std::vector<double> inferno_r_range(8);
	std::vector<double> inferno_g_range(8);
	std::vector<double> inferno_b_range(8);
	std::vector<double> inferno_values(8);

	for (unsigned int i = 0; i < 8; i++) {
		inferno_values[i] = (1.0 / (double)8) * (double)i;
	}

	inferno_r_range[0] = (double)0 / (double)255;
	inferno_g_range[0] = (double)0 / (double)255;
	inferno_b_range[0] = (double)4 / (double)255;

	inferno_r_range[1] = (double)40 / (double)255;
	inferno_g_range[1] = (double)11 / (double)255;
	inferno_b_range[1] = (double)84 / (double)255;

	inferno_r_range[2] = (double)101 / (double)255;
	inferno_g_range[2] = (double)21 / (double)255;
	inferno_b_range[2] = (double)110 / (double)255;

	inferno_r_range[3] = (double)159 / (double)255;
	inferno_g_range[3] = (double)42 / (double)255;
	inferno_b_range[3] = (double)99 / (double)255;

	inferno_r_range[4] = (double)212 / (double)255;
	inferno_g_range[4] = (double)72 / (double)255;
	inferno_b_range[4] = (double)66 / (double)255;

	inferno_r_range[5] = (double)245 / (double)255;
	inferno_g_range[5] = (double)125 / (double)255;
	inferno_b_range[5] = (double)21 / (double)255;

	inferno_r_range[6] = (double)250 / (double)255;
	inferno_g_range[6] = (double)193 / (double)255;
	inferno_b_range[6] = (double)39 / (double)255;

	inferno_r_range[7] = (double)252 / (double)255;
	inferno_g_range[7] = (double)255 / (double)255;
	inferno_b_range[7] = (double)164 / (double)255;

	PopulationHelper* pop = _dws[window_index].population;
	MassPopulation* mass_pop = _dws[window_index].mass_population;
	NdGrid* grid;
	if (pop)
		grid = _dws[window_index].population->getGridOnCard()->getGrid();
	else
		grid = _dws[window_index].mass_population->getGrid();
	
	unsigned int d0 = _dws[window_index].dim0;
	unsigned int d1 = _dws[window_index].dim1;

	double mesh_min_v = grid->getBase()[d0];
	double mesh_max_v = grid->getBase()[d0] + grid->getDims()[d0];
	double mesh_min_h = grid->getBase()[d1];
	double mesh_max_h = grid->getBase()[d1] + grid->getDims()[d1];

	std::vector<unsigned int> cells;
	fptype max_mass = 0.0;
	if (pop) {
		cells = std::vector<unsigned int>(pop->getGridOnCard()->getNumCells());
		for (unsigned int c = 0; c < pop->getNumNeurons(); c++) {
			cells[pop->getSimulation()->getNeuronCellLocations()[pop->getNeuronOffset()+c]]++;
		}

		for (unsigned int i = 0; i < grid->getRes()[d0]; i++) {
			for (unsigned int j = 0; j < grid->getRes()[d1]; j++) {
				std::vector<unsigned int> coords(grid->getNumDimensions());
				coords[d0] = i;
				coords[d1] = j;

				double mass = 0.0;
				std::vector<unsigned int> remaining_dims;
				for (unsigned int d = 0; d < grid->getNumDimensions(); d++) {
					if (d == d0 || d == d1)
						continue;
					remaining_dims.push_back(d);
				}

				calculateMarginal(mass, remaining_dims, coords, pop, mass_pop, cells, grid);

				if (max_mass < mass)
					max_mass = mass;
			}
		}
	}

	
	if (mass_pop) {
		for (unsigned int i = 0; i < grid->getRes()[d0]; i++) {
			for (unsigned int j = 0; j < grid->getRes()[d1]; j++) {
				std::vector<unsigned int> coords(grid->getNumDimensions());
				coords[d0] = i;
				coords[d1] = j;

				double mass = 0.0;
				std::vector<unsigned int> remaining_dims;
				for (unsigned int d = 0; d < grid->getNumDimensions(); d++) {
					if (d == d0 || d == d1)
						continue;
					remaining_dims.push_back(d);
				}

				calculateMarginal(mass, remaining_dims, coords, pop, mass_pop, cells, grid);

				if (max_mass < mass)
					max_mass = mass;
			}
		}
	}
	
	for(unsigned int i = 0; i < grid->getRes()[d0]; i++){
		for(unsigned int j = 0; j < grid->getRes()[d1]; j++) {
			
			// This is the marginal 2D density - so we sum all mass across other dimensions
			std::vector<unsigned int> coords(grid->getNumDimensions());
			coords[d0] = i;
			coords[d1] = j;

			double mass = 0.0;

			std::vector<unsigned int> remaining_dims;
			for (unsigned int d = 0; d < grid->getNumDimensions(); d++) {
				if (d == d0 || d == d1)
					continue;
				remaining_dims.push_back(d);
			}
			
			calculateMarginal(mass, remaining_dims, coords, pop, mass_pop, cells, grid);

			if (pop)
				mass /= max_mass;

			if (mass_pop)
				mass /= max_mass;
			
			unsigned int lower_num = int(mass * 8.0);
			if (lower_num > 6)
				lower_num = 6;

			double green = inferno_g_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num+1] - inferno_values[lower_num])) * (inferno_g_range[lower_num+1] - inferno_g_range[lower_num]));
			double red = inferno_r_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_r_range[lower_num + 1] - inferno_r_range[lower_num]));
			double blue = inferno_b_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_b_range[lower_num + 1] - inferno_b_range[lower_num]));
			
			glColor3f(std::min(1.0,red), std::max(0.0,green), std::max(0.0, blue));
			//glColor3f(mass, mass, mass);
			glVertex2f((i * (2.0 / grid->getRes()[0])) - 1.0, (j * (2.0 / grid->getRes()[1])) - 1.0);
			glVertex2f(((i + 1) * (2.0 / grid->getRes()[0])) - 1.0, (j * (2.0 / grid->getRes()[1])) - 1.0);
			glVertex2f(((i + 1) * (2.0 / grid->getRes()[0])) - 1.0, ((j + 1) * (2.0 / grid->getRes()[1])) - 1.0);
			glVertex2f((i * (2.0 / grid->getRes()[0])) - 1.0, ((j + 1) * (2.0 / grid->getRes()[1])) - 1.0);
		}
	}

	glEnd();

	// Print real time and sim time
	if (SHOW_TEXT) {
		double sim_time = 0.0;

		sim_time = _current_sim_it * _time_step;

		glColor3f(1.0, 1.0, 1.0);
		glRasterPos2f(0.3, 0.9);
		int len, i;
		std::string t = std::string("Sim Time : ") + std::to_string(sim_time);
		const char* c_string = t.c_str();
		len = (int)strlen(c_string);
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
		}

		double h_width = (mesh_max_h - mesh_min_h);
		char buff[32];
		sprintf(buff, "%.*g", 1, h_width);
		double h_step = (double)std::atof(buff) / 10.0;

		std::string s_h_step = std::to_string(h_step);
		s_h_step.pop_back();
		h_step = std::stod(s_h_step);
		double nice_min_h = (double)floor(mesh_min_h / h_step) * h_step;
		double nice_max_h = (double)ceil(mesh_max_h / h_step) * h_step;

		double v_width = (mesh_max_v - mesh_min_v);
		sprintf(buff, "%.*g", 1, v_width);
		double v_step = (double)std::atof(buff) / 10.0;

		std::string s_v_step = std::to_string(v_step);
		s_v_step.pop_back();
		v_step = std::stod(s_v_step);
		double nice_min_v = (double)floor(mesh_min_v / v_step) * v_step;
		double nice_max_v = (double)ceil(mesh_max_v / v_step) * v_step;

		double pos = nice_min_h;
		while (pos < nice_max_h) {
			if (std::abs(pos) < 0.0000000001)
				pos = 0.0;

			glColor3f(1.0, 1.0, 1.0);
			glRasterPos2f(-1.0, 2 * ((pos - (mesh_min_h + ((mesh_max_h - mesh_min_h) / 2.0))) / (mesh_max_h - mesh_min_h)));

			std::stringstream stream;
			stream << std::setprecision(3) << pos;
			t = stream.str();
			c_string = t.c_str();
			len = (int)strlen(c_string);
			for (i = 0; i < len; i++) {
				glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
			}
			pos += h_step;
		}

		pos = nice_min_v;
		while (pos < nice_max_v) {
			if (std::abs(pos) < 0.0000000001)
				pos = 0.0;
			glRasterPos2f(2 * ((pos - (mesh_min_v + ((mesh_max_v - mesh_min_v) / 2.0))) / (mesh_max_v - mesh_min_v)), -1.0);
			std::stringstream stream2;
			stream2 << std::setprecision(3) << pos;
			t = stream2.str();
			c_string = t.c_str();
			len = (int)strlen(c_string);
			for (i = 0; i < len; i++) {
				glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
			}
			pos += (nice_max_v - nice_min_v) / 10;
		}
	}
	
	// **** used for 3D ****
	// glPopMatrix();
	glutSwapBuffers();
	glFlush();

	if(write_frames)
		writeFrame(window_index,_current_sim_it);
}

void Display::writeFrame(unsigned int system, long frame_num){
	//This prevents the images getting padded
 	// when the width multiplied by 3 is not a multiple of 4
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  // width * height * 3
  int nSize = WINDOW_WIDTH * WINDOW_HEIGHT *3;
  // First let's create our buffer, 3 channels per Pixel
  char* dataBuffer = (char*)malloc(nSize*sizeof(char));

  if (!dataBuffer) return;

   // Let's fetch them from the backbuffer
   // We request the pixels in GL_BGR format, thanks to Berzeger for the tip
  //  glReadPixels((GLint)0, (GLint)0,
	// 	(GLint)w, (GLint)h,
	// 	 GL_BGR, GL_UNSIGNED_BYTE, dataBuffer);
	 glReadPixels((GLint)0, (GLint)0,
	 (GLint)WINDOW_WIDTH, (GLint)WINDOW_HEIGHT,
		GL_BGR, GL_UNSIGNED_BYTE, dataBuffer);

		const std::string dirname = std::string("node_") + std::to_string(system);

		if (! boost::filesystem::exists(dirname) ){
			boost::filesystem::create_directory(dirname);
		}

   //Now the file creation
	 std::string filename =  dirname + std::string("/") + std::to_string(frame_num) + std::string(".tga");
   FILE *filePtr = fopen(filename.c_str(), "wb");
   if (!filePtr) return;

   unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
  //  unsigned char header[6] = { w%256,w/256,
	// 		       h%256,h/256,
	// 		       24,0};
	 unsigned char header[6] = { WINDOW_WIDTH %256,WINDOW_WIDTH /256,
						WINDOW_HEIGHT %256,WINDOW_HEIGHT /256,
 						24,0};
   // We write the headers
   fwrite(TGAheader,	sizeof(unsigned char),	12,	filePtr);
   fwrite(header,	sizeof(unsigned char),	6,	filePtr);
   // And finally our image data
   fwrite(dataBuffer,	sizeof(GLubyte),	nSize,	filePtr);
   fclose(filePtr);

   free(dataBuffer);
}

void Display::scene(int width, int height)
{
	glViewport(0, 0, width, height);
	glLoadIdentity();
}

void Display::init() const {
	
}

void Display::update() {
}

void Display::updateDisplay(long current_sim_it) {
	if (Display::getInstance()->_nodes_to_display.size() == 0)
		return;

	int time;
	time = glutGet(GLUT_ELAPSED_TIME);
	Display::getInstance()->_current_sim_it = current_sim_it;
	lastTime = time;
	std::this_thread::sleep_for(std::chrono::milliseconds(1));
	for (unsigned int id = 0; id < _nodes_to_display.size(); id++) {
		if(!glutGetWindow())
			continue;
		glutSetWindow(_dws[_nodes_to_display[id]]._window_index);
		glutPostRedisplay();
	}
#ifndef USING_APPLE_GLUT
	glutMainLoopEvent();
#else
	glutCheckLoop();
#endif

}

void Display::shutdown() const {
#ifndef USING_APPLE_GLUT // Hack to avoid issues with OSX glut version
	glutExit();
#endif

	// Nice new line if we quit early.
	std::cout << "\n";
}
void Display::setDisplayNodes(std::vector<unsigned int> nodes_to_display) const {
	Display::getInstance()->_nodes_to_display = nodes_to_display;
}

void Display::animate(bool _write_frames, std::vector<unsigned int> display_nodes, double time_step) const {
	Display::getInstance()->_nodes_to_display = display_nodes;
	animate(_write_frames, time_step);
}

void Display::animate(bool _write_frames,  double time_step) const{

	if (Display::getInstance()->_nodes_to_display.size() == 0)
		return;

	Display::getInstance()->write_frames = _write_frames;
	Display::getInstance()->_time_step = time_step;

	char* arv[] = {"FastMC"};
	int count = 1;
	glutInit(&count, arv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(0, 0);
	glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF);

	for (unsigned int id = 0; id < Display::getInstance()->_nodes_to_display.size(); id++) {
		Display::getInstance()->_dws[Display::getInstance()->_nodes_to_display[id]]._window_index = glutCreateWindow("FastMC");
		glutDisplayFunc(Display::stat_display);
		glutReshapeFunc(Display::stat_scene);
		glutIdleFunc(Display::stat_update);
	}

	atexit(Display::stat_shutdown);
// glutSetOption is not available in OSX glut - on other OSs (using freeglut), this allows us to keep running the simulation 
// even though the window is closed
// I don't know what will happen on OSX because I don't live and work in Shoreditch. 
#ifndef USING_APPLE_GLUT
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
#endif
	init();
}

void Display::processDraw(void) {
}
