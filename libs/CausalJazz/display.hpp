#ifndef GRID_DISPLAY
#define GRID_DISPLAY

#define LINUX // This #define is used in glew.c so it does not require installation
#include "GL/glew.h"
#include <GL/freeglut.h>


#include <string>
#include <chrono>
#include <mutex>
#include <map>

#include "MassPopulation.cuh"

class DisplayWindow{
public:
    MassPopulation* mass_population;
    int _window_index;

    double max_mass;
    double min_mass;

    unsigned int dim0;
    unsigned int dim1;

    double width;
    double height;
};

class Display{

public:

  static Display* getInstance() {
    if (!disp) {
      disp = new Display();
    }

    return disp;
  }

  void display(void);
  void scene(int width, int height);
  void init() const;
  void update();
  void shutdown() const;
  void setDisplayNodes(std::vector<unsigned int> nodes_to_display) const ;
  void animate(bool, double time_step) const;
  void animate(bool, std::vector<unsigned int>, double time_step) const;
  void processDraw(void);

  void calculateMarginal(double& total, std::vector<unsigned int> remaining_dims, 
      std::vector<unsigned int> coords, MassPopulation* mass_pop, 
      std::vector<unsigned int>& cells, NdGrid* grid);

  static void stat_display(void) {
    disp->display();
  }
  static void stat_scene(int width, int height) {
    disp->scene(width,height);
  }
  static void stat_update(void){
    disp->update();
  }
  static void stat_shutdown(void){
    disp->shutdown();
  }

  unsigned int addMassPopulation(unsigned int nid, MassPopulation* pop);

  void updateDisplay(long current_sim_it);

private:

  bool write_frames;
  void writeFrame(unsigned int system, long frame_num);

  static Display* disp;

  Display();
  ~Display();

  long _current_sim_it;
  double _time_step;

  int lastTime;

  std::vector<unsigned int> _nodes_to_display;

  std::chrono::milliseconds start_time;

  std::map<unsigned int, DisplayWindow> _dws;

  bool upPressed;
  bool downPressed;
  bool leftPressed;
  bool rightPressed;
  bool pgupPressed;
  bool pgdnPressed;
};


#endif
