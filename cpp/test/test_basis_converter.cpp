#include "sando_cpp/basis_converter.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
using std::string; using std::vector;
static vector<double> nums(const string&r){vector<double>v;std::istringstream is(r);double x;while(is>>x)v.push_back(x);return v;}
int main(int argc,char**argv){
  const char*path=(argc>1)?argv[1]:"golden/basis_converter_cases.txt";
  std::ifstream f(path); if(!f){std::printf("CANNOT OPEN\n");return 2;}
  sando::BasisConverter bc; std::map<string,vector<double>> cur; string line; int nfail=0; double gmax=0; const double TOL=1e-9;
  auto chk=[&](const std::vector<Eigen::Matrix4d>&v,const char*k){double e=0;auto&g=cur[k];for(size_t m=0;m<v.size();m++)for(int i=0;i<4;i++)for(int j=0;j<4;j++)e=std::max(e,std::fabs(v[m](i,j)-g[(m*4+i)*4+j]));gmax=std::max(gmax,e);if(e>=TOL)nfail++;};
  auto process=[&](){
    int num=(int)cur["NUM"][0];
    chk(bc.get_minvo_pos_converters(num),"MVP"); chk(bc.get_bezier_pos_converters(num),"BEP"); chk(bc.get_A_bspline(num),"ABS");
    {auto v=bc.get_minvo_vel_converters(num);double e=0;auto&g=cur["MVV"];for(size_t m=0;m<v.size();m++)for(int i=0;i<3;i++)for(int j=0;j<3;j++)e=std::max(e,std::fabs(v[m](i,j)-g[(m*3+i)*3+j]));gmax=std::max(gmax,e);if(e>=TOL)nfail++;}
    {auto v=bc.get_minvo_accel_converters(num);double e=0;auto&g=cur["MVA"];for(size_t m=0;m<v.size();m++)for(int i=0;i<2;i++)for(int j=0;j<2;j++)e=std::max(e,std::fabs(v[m](i,j)-g[(m*2+i)*2+j]));gmax=std::max(gmax,e);if(e>=TOL)nfail++;}
  };
  while(std::getline(f,line)){if(line=="CASE"){cur.clear();continue;}if(line=="END"){process();continue;}std::istringstream is(line);string k;is>>k;string r;std::getline(is,r);cur[k]=nums(r);}
  std::printf("basis_converter golden: %d fail, global_max_err=%.3e (tol=%.0e)\n%s\n",nfail,gmax,TOL,nfail==0?"ALL PASS":"FAILED");
  return nfail==0?0:1;
}
