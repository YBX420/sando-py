// Golden verification for bspline.hpp vs bspline.py.
#include "sando_cpp/bspline.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

using std::string; using std::vector;
static vector<double> nums(const string& r){vector<double> v;std::istringstream is(r);double x;while(is>>x)v.push_back(x);return v;}

int main(int argc, char** argv){
  const char* path=(argc>1)?argv[1]:"golden/bspline_cases.txt";
  std::ifstream f(path); if(!f){std::printf("CANNOT OPEN %s\n",path);return 2;}
  std::map<string,vector<double>> cur; string kind,line;
  int ncase=0,nfail=0; double gmax=0; const double TOL=1e-7;

  auto process=[&](){
    double e=0;
    if(kind=="SPLINE"){
      int p=(int)cur["P"][0], Nc=(int)cur["NCTRL"][0]; double dt=cur["DT"][0];
      Eigen::MatrixXd ctrl(Nc,3);
      for(int i=0;i<Nc;i++) for(int j=0;j<3;j++) ctrl(i,j)=cur["CTRL"][i*3+j];
      sando::UniformBSpline bs(ctrl,p,dt);
      e=std::max(e,std::fabs(bs.t_start-cur["TSTART"][0]));
      e=std::max(e,std::fabs(bs.t_end-cur["TEND"][0]));
      int K=(int)cur["K"][0];
      for(int k=0;k<K;k++){
        double t=cur["TS"][k];
        auto a0=bs.eval(t), a1=bs.eval_deriv(t,1), a2=bs.eval_deriv(t,2);
        for(int j=0;j<3;j++){
          e=std::max(e,std::fabs(a0(j)-cur["E0"][k*3+j]));
          e=std::max(e,std::fabs(a1(j)-cur["E1"][k*3+j]));
          e=std::max(e,std::fabs(a2(j)-cur["E2"][k*3+j]));
        }
      }
      int NB=(int)cur["NBN"][0];
      for(int q=0;q<NB;q++){
        auto kn=bs.nonzero_basis(cur["NBT"][q]);
        e=std::max(e,std::fabs((double)kn.first-cur["NBSPAN"][q]));
        for(int c=0;c<p+1;c++) e=std::max(e,std::fabs(kn.second(c)-cur["NBVAL"][q*(p+1)+c]));
      }
    } else if(kind=="FIT"){
      int p=(int)cur["P"][0], numc=(int)cur["NUMCTRL"][0], Mp=(int)cur["M"][0]; double dt=cur["DT"][0];
      bool clamp=(int)cur["CLAMP"][0]!=0;
      Eigen::MatrixXd pathm(Mp,3);
      for(int i=0;i<Mp;i++) for(int j=0;j<3;j++) pathm(i,j)=cur["PATH"][i*3+j];
      auto bs=sando::UniformBSpline::fit_path(pathm,numc,p,dt,clamp);
      for(int i=0;i<numc;i++) for(int j=0;j<3;j++)
        e=std::max(e,std::fabs(bs.ctrl(i,j)-cur["CTRLOUT"][i*3+j]));
      int K=(int)cur["K"][0];
      for(int k=0;k<K;k++){ auto ev=bs.eval(cur["TS"][k]);
        for(int j=0;j<3;j++) e=std::max(e,std::fabs(ev(j)-cur["EV"][k*3+j])); }
    }
    gmax=std::max(gmax,e); bool ok=e<TOL;
    std::printf("  %-7s : max_err=%.3e  %s\n",kind.c_str(),e,ok?"PASS":"FAIL");
    if(!ok)nfail++; ncase++;
  };

  while(std::getline(f,line)){
    if(line=="SPLINE"||line=="FIT"){cur.clear();kind=line;continue;}
    if(line=="END"){process();continue;}
    std::istringstream is(line);string key;is>>key;string rest;std::getline(is,rest);cur[key]=nums(rest);
  }
  std::printf("\nbspline golden: %d cases, %d fail, global_max_err=%.3e (tol=%.0e)\n",ncase,nfail,gmax,TOL);
  std::printf("%s\n", nfail==0?"ALL PASS (C++ reproduces Python)":"FAILED");
  return nfail==0?0:1;
}
