#include <vector>
#include <iostream>
#include <fstream>
//#include <cmath> 
#include <string>
#include <sstream>
#include <map>
#include <iterator>
#include <parallel/numeric>
#include <parallel/algorithm>

//#define _USE_MATH_DEFINES // for C++  
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
                    if(vstrings[0]=="h"|vstrings[0]=="V"|vstrings[0]=="phis"){
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
                    if(vstrings[0]=="Npar"|vstrings[0]=="A"){
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

void dump_to_file(std::string& path, std::vector<double>& data){
    std::filebuf fb;
	fb.open(path, std::ios::out);
	std::ostream os(&fb);
	for (int i = 0; i<data.size(); ++i) {
		os << data[i] << '\n';
	}
	fb.close();
};


double covariance(std::vector<double>& a, std::vector<double>& b, double meanA, double meanB){
    std::vector<double> temp(a.size(),0);
#pragma omp parallel for
    for(int i = 0;i<a.size();++i){
        temp[i] = (a[i]-meanA)*(b[i]-meanB);
    }
    return __gnu_parallel::accumulate(temp.begin(),temp.end(),0.0)/a.size();
}


double max_para(std::vector<double>& x){
    return x[std::distance(x.begin(),__gnu_parallel::max_element(x.begin(),x.end()))];
}

double min_para(std::vector<double>& x){
    return x[std::distance(x.begin(),__gnu_parallel::min_element(x.begin(),x.end()))];
}

// the function do  the binning of the particles.
// "data" is the vector of input data
// "counts" is the vector stores the number of particles in each bin.
// "bin_index" is the vector stores the index of bin where ith particle is located.
// "bin_size" is the size(width) of the bin.
// "n_bin_each_side" is the number of bins on each side from center of the bunch.
void binning(std::vector<double>& data, std::vector<unsigned int>& counts, std::vector<unsigned int>& bin_index, double data_mean, double bin_size, double n_bins_each_side){
    double bin_max = data_mean+bin_size*n_bins_each_side; // center of the right most bin,
    double bin_min = data_mean-bin_size*n_bins_each_side; // center of the left most bin,
    unsigned int tot_bins = 2*n_bins_each_side+1;
    double left = bin_min-data_mean-bin_size*0.5; // left edge of the current range where the particle is located,
    double right = bin_max+bin_size*0.5; // rigth edge of the current range where the particle is located.
    for(int i = 0;i<data.size();++i){
        double left = data_mean-bin_size*0.5; // left edge of the current range where the particle is located,
        double right = bin_max+bin_size*0.5; // rigth edge of the current range where the particle is located.
        while(right-left>bin_size){
            if(data[i]>(left+right)*0.5){
                left = (left+right-bin_size)*0.5;
            }
            else{
                right = (left+right+bin_size)*0.5;
            }
        }
        bin_index[i] = n_bins_each_side+int((left+right)*0.5/bin_size);
        counts[bin_index[i]] += 1;
    }
}
