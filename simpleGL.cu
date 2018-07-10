// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <iterator>
#include <map>
#include <vector>
#include <iterator>
// #include <parallel/numeric>
// #include <parallel/algorithm>

#include <curand.h>
#include <curand_kernel.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include "helper_gl.h"
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     5 //ms


////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 728;
const unsigned int window_height = 728;

double c = 3e8;
double E0 = 938e6; //rest energy of proton
double q = 1.6e-19; // charge of proton

unsigned int n_per_show = 2;
unsigned int n_turns = 1e8;
unsigned int finished = 0;

double Qx = 3.666;
double nux = 2.0*3.1415926*Qx;
double sinnux = sin(nux);
double cosnux = cos(nux);
double Qy = 3.7;
double nuy = 2.0*3.1415926*Qy;
double sinnuy = sin(nuy);
double cosnuy = cos(nuy);
double S = 0.0;
double bt = 1.0;
double v1 = 1;//used to manually control the voltage.
double v2 = 0;
double v3 = 0;
double glob_time = 0;
double ramp_time = 1e8;
int Nturns = 10000;

// ring parameters
double R = 610.1754; // radius of the ring
double GMTSQ = 552.25; // gamma_t square
double Gamma = 293;
double Ek = Gamma*E0;
double f0 = c/(2*M_PI*R);
double T0 = 1/f0;
double omg0 = 2*M_PI*f0;
double eta = 1/GMTSQ-1/(Gamma*Gamma);
// RF parameters
int nRF = 1;
double ht = 7200; // harmonic number of target RF
int targetRF_on = 0;
int h1 = 360; // harmonic number of fundamental RF
int h2 = 7200; // harmonic number of fundamental RF

int hm = 360;
double V1 = 2e6; // Voltage of fundamental RF
double V1i = 2e6;
double V1f = 3e6;
double V2 = 2e6; // Voltage of fundamental RF
double V2i = 2e6;
double V2f = 3e6;
unsigned int it1 = 0;
unsigned int it2 = 0;
unsigned int ft1 = 1e4;
unsigned int ft2 = 1e4;

int tn_btwn_i_f = 1e5;
double Vepsilon = 0.5; // modulation depth
double Vm = Vepsilon*V1;
double phis = M_PI; // synchronous phase of fundamental RF
double Qs = sqrt(h1*V1i*eta*abs(cos(phis))/(2*M_PI*Ek));
double Mum = 2.1; // modulation tune factor
double Qm = Mum*Qs; // modulation tune

// bunch parameters
unsigned int Npar = 1<<15;
unsigned int newNpar = 0;//number of particles after applying the filter after the whole tracking process.
double A = 0.8; // 95% emittance, in unit of eV.S
double epsln = A/6;
double delta_hat = sqrt(epsln)*sqrt(omg0/(M_PI*Ek))*std::pow((h1*V1*abs(cos(phis))/(2*M_PI*Ek*eta)),0.25);
double t_hat = delta_hat/Qs*eta/omg0;
double ts = phis/h1/omg0;
unsigned int initial = 0;
int cT = 0; // current turn
double4 conditions={-1e6,1e6,-1e6,1e6};

// vbo variables
GLuint vbo;
GLuint vbox;
struct cudaGraphicsResource *cuda_vbo_resource;
struct cudaGraphicsResource *cuda_vbo_resourcex;
void *d_vbo_buffer = NULL;
void *d_vbo_bufferx = NULL;
float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -5;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float GFLOPS = 0;        // GFLOPS
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)
class inputPara{
public:
    double c = 3e8;
    double E0 = 938e6; //rest energy of proton
    double q = 1.6e-19; // charge of proton
    int cT = 0; // current turn

    // general parameters
    std::map<std::string,unsigned int> generalPara;
   
    // ring parameters
    std::map<std::string,double> ringPara;
    // RF parameters
    std::map<std::string,std::vector<double>> rfPara;
    
    // Modulation
    std::map<std::string,double> modPara;

    // bunch parameters

    std::map<std::string,double> bunchPara;
    
    inputPara(){
        c = 3e8;
        E0 = 938e6; //rest energy of proton
        q = 1.6e-19; // charge of proton
        cT = 0; // current turn
    
        // general parameters
        generalPara={
            {"n_turns",1e8},
            {"n_per_show",2}
        };
       
        // ring parameters
        ringPara={
        {"R",610.1754}, // radius of the ring
        {"GMTSQ",552.25}, // gamma_t square
        {"Gamma",293},
        {"Ek",0},//ringPara["Gamma"]*E0},
        {"f0",0},//c/(2*M_PI*ringPara["R"])},
        {"T0",0},// 1/ringPara["f0"]},
        {"omg0",0},//2*M_PI*ringPara["f0"]},
        {"eta",0},//1/ringPara["GMTSQ"]-1/(ringPara["Gamma"]*ringPara["Gamma"])}
        {"nRF",1}
        };
        // RF parameters
        rfPara={
            {"h",std::vector<double>(1)},
            {"V",std::vector<double>(2e6)},
            {"Vf",std::vector<double>(2e6)},
            {"it",std::vector<double>(0)}, // initial turn number
            {"ft",std::vector<double>(1e4)}, // final turn number
            {"tn_btwn_i_f",std::vector<double>(1e5)},
            {"phis",std::vector<double>(M_PI)},
            {"Qs",std::vector<double>(0)}// sqrt(h[0]*V[0]*eta*abs(cos(phis[0]))/(2*M_PI*Ek));
        };
        
        // Modulation
        modPara={
        {"Vepsilon",0.5}, // modulation depth
        {"Mum",2.1}, // modulation tune factor
        };
    
        // bunch parameters
    
        bunchPara={
            {"Npar",1e6},
            {"A",0.8},
            {"epsln",0},//bunchPara["A"]/6},
            {"delta_hat",0},//sqrt(epsln)*sqrt(ringPara["omg0"]/(M_PI*ringPara["Ek"]))*std::pow((rfPara["h"][0]*rfPara["V"][0]*abs(cos(rfPara["phis"][0]))/(2*M_PI*ringPara["Ek"]*ringPara["eta"])),0.25)},
            {"t_hat",0}//bunchPara["delta_hat"]/rfPara["Qs"]*ringPara["eta"]/ringPara["omg0"]
        };
    };
    int read(std::string path){
        // Object to write in file
        std::ifstream fin;
        std::cout<<"Created file Stream"<<std::endl;
        // Opening file in append mode
        fin.open(path, std::ios::in);
        std::cout<<"Opened file"<<std::endl;

        std::string temp;
        // Read the object's data in file
        while(std::getline(fin,temp)){
            if(temp.size()>1){
                std::stringstream ss(temp);
                std::istream_iterator<std::string> begin(ss);
                std::istream_iterator<std::string> end;
                std::vector<std::string> vstrings(begin, end);
                if(vstrings.size()>1){
                    if(vstrings[0]=="nRF"){
                        for(auto& x:rfPara){
                            x.second.resize(stoi(vstrings[1]));//resize the vectors that store voltage, phase and impendance info
                        }
                    }
                    if(vstrings[0]=="it"|vstrings[0]=="ft"|vstrings[0]=="h"|vstrings[0]=="V"|vstrings[0]=="Vf"|vstrings[0]=="tn_btwn_i_f"|vstrings[0]=="phis"){
                        for(unsigned int i=0;i<rfPara[vstrings[0]].size();++i){
                            rfPara[vstrings[0]][i]=stod(vstrings[i+1]);
                        }
                    }
                    if(vstrings[0]=="n_turns"|vstrings[0]=="n_per_show"){
                        generalPara[vstrings[0]]=int(stod(vstrings[1]));
                    }
                    if(vstrings[0]=="R"|vstrings[0]=="GMTSQ"|vstrings[0]=="Gamma"|vstrings[0]=="nRF"){
                        ringPara[vstrings[0]]=stod(vstrings[1]);
                    }
                    if(vstrings[0]=="Vepsilon"|vstrings[0]=="Mum"){
                        modPara[vstrings[0]]=stod(vstrings[1]);
                    }
                    if(vstrings[0]=="Npar"|vstrings[0]=="A"|vstrings[0]=="xmin"|vstrings[0]=="xmax"|vstrings[0]=="ymin"|vstrings[0]=="ymax"){
                        bunchPara[vstrings[0]]=stod(vstrings[1]);
                    }
                }
            }
        }
        ringPara["Ek"]=ringPara["Gamma"]*E0;
        ringPara["f0"]=c/(2*M_PI*ringPara["R"]);
        ringPara["T0"]=1/ringPara["f0"];
        ringPara["omg0"]=2*M_PI*ringPara["f0"];
        ringPara["eta"]=1/ringPara["GMTSQ"]-1/(ringPara["Gamma"]*ringPara["Gamma"]);
        rfPara["Qs"][0]=sqrt(rfPara["h"][0]*rfPara["V"][0]*ringPara["eta"]*abs(cos(rfPara["phis"][0]/180*M_PI))/(2*M_PI*ringPara["Ek"]));
        bunchPara["epsln"]=bunchPara["A"]/6.0;
        bunchPara["delta_hat"]=sqrt(bunchPara["epsln"])*sqrt(ringPara["omg0"]/(M_PI*ringPara["Ek"]))*std::pow((rfPara["h"][0]*rfPara["V"][0]*abs(cos(rfPara["phis"][0]/180*M_PI))/(2*M_PI*ringPara["Ek"]*ringPara["eta"])),0.25);
        bunchPara["t_hat"]=bunchPara["delta_hat"]/rfPara["Qs"][0]*ringPara["eta"]/ringPara["omg0"];
        return 0;
    };
    int printout(){
        std::cout<<"General Parameters:"<<std::endl;
        for(auto& x:generalPara){
            std::cout<<x.first<<"="<<x.second<<std::endl;
        }
        std::cout<<"Ring Parameters:"<<std::endl;
        for(auto& x:ringPara){
            std::cout<<x.first<<"="<<x.second<<std::endl;
        }
        std::cout<<"Rf Parameters:"<<std::endl;
        for(auto& x:rfPara){
            std::cout<<x.first;
            for(auto&data:x.second){
                std::cout<<":"<<data;
            }
            std::cout<<std::endl;
        }
        std::cout<<"Modulation Parameters:"<<std::endl;
        for(auto& x:modPara){
            std::cout<<x.first<<"="<<x.second<<std::endl;
        }
        std::cout<<"Bunch Parameters:"<<std::endl;
        for(auto& x:bunchPara){
            std::cout<<x.first<<"="<<x.second<<std::endl;
        }
        return 0;
    };
};
int getInputs(inputPara &inputs){
    std::cout<<Qs<<std::endl;
    std::cout<<T0<<std::endl;
    std::cout<<phis<<std::endl;

    //General
    n_turns = inputs.generalPara["n_turns"];
    n_per_show = inputs.generalPara["n_per_show"];
    //Ring
    R = inputs.ringPara["R"];
    GMTSQ = inputs.ringPara["GMTSQ"];
    Gamma = inputs.ringPara["Gamma"];
    Ek = inputs.ringPara["Ek"];
    f0 = inputs.ringPara["f0"];
    T0 = inputs.ringPara["T0"];
    omg0 = inputs.ringPara["omg0"];
    eta = inputs.ringPara["eta"];
    nRF = inputs.ringPara["nRF"];
    //RF
    h1 = inputs.rfPara["h"][0];
    V1 = inputs.rfPara["V"][0];
    V1f = inputs.rfPara["Vf"][0];
    it1 = inputs.rfPara["it"][0];
    ft1 = inputs.rfPara["ft"][0];
    it2 = inputs.rfPara["it"][1];
    ft2 = inputs.rfPara["ft"][1];

    tn_btwn_i_f = inputs.rfPara["tn_btwn_i_f"][0];
    phis = inputs.rfPara["phis"][0]/180*M_PI;
    Qs = inputs.rfPara["Qs"][0];
    //Modulation
    Vepsilon = inputs.modPara["Vepsilon"];
    Mum = inputs.modPara["Mum"];
    Qm = Mum*Qs;
    //Bunch
    Npar = inputs.bunchPara["Npar"];
    A = inputs.bunchPara["A"];
    epsln = A/6;
    delta_hat = inputs.bunchPara["delta_hat"];
    t_hat = inputs.bunchPara["t_hat"];
    conditions.x = inputs.bunchPara["xmin"];
    conditions.y = inputs.bunchPara["xmax"];
    conditions.z = inputs.bunchPara["ymin"];
    conditions.w = inputs.bunchPara["ymax"];
    std::cout<<Qs<<std::endl;
    std::cout<<T0<<std::endl;
    std::cout<<phis<<std::endl;
    return 0;
}

// Post process//
void Filter(double2 *oldCords, double2 *newCords,double4 conditions,unsigned int &Npar,unsigned int &newNpar){
    for(unsigned int i= 0;i<Npar;++i){
        if(oldCords[i].x>=conditions.x&oldCords[i].x<=conditions.y&oldCords[i].y>=conditions.z&oldCords[i].y<conditions.w){
            newCords[newNpar].x=oldCords[i].x;
            newCords[newNpar].y=oldCords[i].y;
            newNpar++;
        }
    }
}

double Emittance(double2 *cords, unsigned int N){
    std::cout<<"New Npar:"<<N<<std::endl;
    // get the means 
    std::vector<double> cordx(N,0.0);
    std::vector<double> cordy(N,0.0);

    std::cout<<"New Npar:"<<cordx.size()<<std::endl;

    double meanx;
    double meany;
#pragma omp parallel for simd
    for(unsigned int i = 0;i<N;++i){
        cordx[i] = cords[i].x;
        cordy[i] = cords[i].y;
    }
    meanx = std::accumulate(cordx.begin(),cordx.end(),0.0)/Npar;
    meany = std::accumulate(cordy.begin(),cordy.end(),0.0)/Npar;
    std::cout<<meanx<<","<<meany<<std::endl;
    // get the covariance
    std::vector<double> temp(Npar,0);
    double varx;
    double vary;
    double covarxy;
#pragma omp parallel for simd
    for(int i = 0;i<N;++i){
        temp[i] = (cordx[i]-meanx)*(cordx[i]-meanx);
    }
    varx = std::accumulate(temp.begin(),temp.end(),0.0)/Npar;
#pragma omp parallel for simd
    for(int i = 0;i<N;++i){
        temp[i] = (cordy[i]-meany)*(cordy[i]-meany);
    }
    vary = std::accumulate(temp.begin(),temp.end(),0.0)/Npar;
#pragma omp parallel for simd
    for(int i = 0;i<N;++i){
        temp[i] = (cordx[i]-meanx)*(cordy[i]-meany);
    }
    covarxy = std::accumulate(temp.begin(),temp.end(),0.0)/Npar;
    std::cout<<varx<<","<<vary<<","<<covarxy<<std::endl;

    // Emittance
    double emittance;
    emittance = sqrt(varx*vary-covarxy*covarxy);
    std::cout<<emittance<<std::endl;
    return emittance;
}
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file,inputPara input);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource,struct cudaGraphicsResource **vbo_resourcex);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void initCurand(curandState *state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, 0, 0, &state[idx]);
}
__global__ void init_kernel(double2 *pos, double2 *posx,unsigned int Npar,
    double sigdelta, // sigma delta
    double sigt, // sigma t
    double Ek,   // mean kinetic energy
    double ts)   // synchronous time.
{
    unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
    /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
    curandState_t local_state;

  /* we have to initialize the state */
    curand_init(n, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &local_state);

    double delta = curand_normal_double(&local_state)*sigdelta;
    double t = curand_normal_double(&local_state)*sigt;

    // write output vertex
    pos[n] = make_double2(t,delta);
    posx[n] = make_double2(t,delta);
}

__global__ void simple_vbo_kernel(
    double2 *pos, double2 *posx,
    double eta,
    double Ek,
    double V1,
    double V1f,
    double V2,
    double V2f,
    unsigned int it1,
    unsigned int ft1,
    unsigned int it2,
    unsigned int ft2,
    double Vepsilon,
    double Qm,
    double omg0,
    double phis,
    int h1,
    double T0,
    int cT, // Current turn
    unsigned int n_per_show
    )
{
    unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
    double t = pos[n].x;  // corresponding to time.  
    double delta = pos[n].y;   // corresponding to Energy.  
    double V;
    for(unsigned int i = 0;i<n_per_show;++i){
        V = V1+(V1f-V1)/(ft1-it1)*(cT+i);
        delta += V*(1+Vepsilon*sin(Qm*omg0*(t+(cT+i)*T0)))*(sin(h1*omg0*t+phis)-sin(phis))/Ek;
        t += T0*eta*delta;
    }

    // write output vertex
    
    pos[n] = make_double2(t,delta);
    posx[n] = make_double2(t*1e8,delta*1000);
}


void launch_kernel(double2 *pos, double2 *posx,unsigned int Npar,double nu,double S,double bt,int cT)
{
    // execute the kernel
    dim3 block(1024, 1, 1);
    dim3 grid(Npar/ block.x>1?Npar/ block.x:1, 1, 1);

    simple_vbo_kernel<<<grid, block>>>(pos,posx,eta,Ek,V1,V1f,V2,V2f,it1,ft1,it2,ft2,Vepsilon,Qm,omg0,phis,h1,T0,cT,n_per_show);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);
        GFLOPS = 56.0f*Npar*n_per_show/(sdkGetAverageTimerValue(&timer) / 1000.f)/1000000000;
        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "#Particles:%4d; #Turns: %4d; GFLOPS: %4.4f, Time per turn (us): %4.4f", Npar,frameCount*n_per_show,GFLOPS,sdkGetAverageTimerValue(&timer)/n_per_show*1000);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.001, 1000.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file, inputPara input)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return false;
    }

    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif
    // create VBO
    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    createVBO(&vbox, &cuda_vbo_resourcex, cudaGraphicsMapFlagsWriteDiscard);
    // run the cuda part
    //runCuda(&cuda_vbo_resource,&cuda_vbo_resourcex);
    // start rendering mainloop
    glutMainLoop();
    return true; 
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource,
    struct cudaGraphicsResource **vbo_resourcex)
{
    // map OpenGL buffer object for writing from CUDA
    double2 *dptr;
    double2 *dptrx;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource,0));
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resourcex,0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr,  &num_bytes, *vbo_resource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptrx, &num_bytes,*vbo_resourcex));
 
    dim3 block(1024, 1, 1);
    dim3 grid(Npar/ block.x>1?Npar/ block.x:1, 1, 1);
    if (initial == 0){
        double2 *test;
        init_kernel<<<grid,block>>>(dptr,dptrx,Npar,delta_hat,t_hat,Ek,ts);
        test =(double2*) malloc(Npar*sizeof(double2));
        cudaMemcpy(test, dptr, Npar*sizeof(double2), cudaMemcpyDeviceToHost);
        std::cout<<test[0].y<<std::endl;
        initial =1;
    }
    launch_kernel(dptr, dptrx, Npar, nux, S, bt,cT);
    if (finished == 1){
        double2 *oldCords;
        double2 *newCords;
        double emittance = 0;
        char wait;
        oldCords =(double2*) malloc(Npar * sizeof(double2));
        newCords =(double2*) malloc(Npar * sizeof(double2));
        cudaMemcpy(oldCords, dptr, Npar*sizeof(double2), cudaMemcpyDeviceToHost);
        Filter(oldCords,newCords,conditions,Npar,newNpar);
        std::cout<<"New number of particles:"<<newNpar<<std::endl;
        emittance = Emittance(newCords,newNpar);
        std::cout<<"Particle Lost rate (%):"<<double(Npar-newNpar)/double(Npar)*100<<std::endl;
        std::cout<<"rms Emittance is:"<<emittance*Ek*M_PI*6<<std::endl;
        std::cin>>wait;
        exit(0);
    }
    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resourcex, 0));
}


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = Npar * 4 * sizeof(double);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    //checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    bt-=0.0001;
    runCuda(&cuda_vbo_resource,&cuda_vbo_resourcex);
    cT+=n_per_show;
    g_fAnim += 0.01f;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    glPointSize(4.0);
    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbox);
    glVertexPointer(2, GL_DOUBLE, 2*sizeof(double), 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0, 0.0, 1.0);
    glDrawArrays(GL_POINTS, 0, Npar);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    sdkStopTimer(&timer);
    computeFPS();
    glob_time+=n_per_show;
    // if (glob_time > ramp_time){
    //     exit(0);
    // }
    if (glob_time > n_turns){
        finished = 1;
    }
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
            
        case (32):
            rotate_x = 0;
            rotate_y = 0;
            
            break;
        case ('r'):
            initial = 0;
            break;
        case ('a'): // left arrow key reduce sextuple strength
            S-=0.1;
            break;
        case ('d'):// right arrow key reduce sextuple strength
            S+=0.1;
            break;
        case ('w'):// up arrow key increase tune
            Qx+=0.01;
            nux = 2.0*3.1415926*Qx;
            break;
        case ('s'):// down arrow key decrease tune
            Qx-=0.01;
            nux = 2.0*3.1415926*Qx;
            break;
        case ('f'):// f key increase second order harmonic
            v2 +=0.01;
            std::cout<<"V2="<<v2<<std::endl;
            break; 
        case ('v'):// g key decrease second order harmonic
            v2 -=0.01;
            std::cout<<"V2="<<v2<<std::endl;
            break; 
        case ('g'):// f key increase second order harmonic
            v1 +=0.01;
            std::cout<<"V1="<<v1<<std::endl;
            break; 
        case ('b'):// g key decrease second order harmonic
            v1 -=0.01;
            std::cout<<"V1="<<v1<<std::endl;
            break;
        case ('h'):// f key increase second order harmonic
            v3 +=0.01;
            std::cout<<"V3="<<v3<<std::endl;
            break; 
        case ('n'):// g key decrease second order harmonic
            v3 -=0.01;
            std::cout<<"V3="<<v3<<std::endl;
            break; 
            
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    char *ref_file = NULL;
    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);
    inputPara input1;
    std::string path = "input.txt";
    input1.read(path);
    std::cout<<"Read input parameters."<<std::endl;
    input1.printout();
    getInputs(input1);
    
    printf("\n");
    std::cout<<"t_hat is:"<<t_hat<<std::endl;
    std::cout<<"delta_hat is:"<<delta_hat<<std::endl;
    std::cout<<"Ek is:"<<Ek<<std::endl;
    std::cout<<"T0 is:"<<T0<<std::endl;
    std::cout<<"Eta is:"<<eta<<std::endl;

    curandState *devState;
    cudaMalloc((void**)&devState, Npar * sizeof(curandState));
    dim3 block(1024, 1, 1);
    dim3 grid(Npar/ block.x>1?Npar/ block.x:1, 1, 1);
    initCurand<<<grid,block>>>(devState, 1);
    cudaDeviceSynchronize();

    runTest(argc, argv, ref_file,input1);
    
    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");

    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}