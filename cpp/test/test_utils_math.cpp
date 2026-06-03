// Golden verification for utils_math.hpp (+ types_minor.hpp compile) vs utils.py.
#include "sando_cpp/utils_math.hpp"
#include "sando_cpp/types_minor.hpp"
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
  const char* path=(argc>1)?argv[1]:"golden/utils_math_cases.txt";
  std::ifstream f(path); if(!f){std::printf("CANNOT OPEN %s\n",path);return 2;}
  // compile-smoke types_minor
  sando::StateDeriv sd; sando::Polytope poly; (void)sd; (void)poly.contains(Eigen::Vector3d(0,0,0));

  std::map<string,vector<double>> cur; string kind,line;
  int nfail=0; double gmax=0; const double TOL=1e-9;
  auto v3=[&](const char*k){return Eigen::Vector3d(cur[k][0],cur[k][1],cur[k][2]);};
  auto upd=[&](double e){gmax=std::max(gmax,e);};

  auto blk=[&](){
    double e=0;
    if(kind=="SAT"){ auto o=sando::saturate(v3("V"),cur["VMAX"][0]); for(int j=0;j<3;j++)e=std::max(e,std::fabs(o(j)-cur["OUT"][j])); }
    else if(kind=="PSPH"){ auto o=sando::project_point_to_sphere(v3("P1"),v3("P2"),cur["R"][0]); for(int j=0;j<3;j++)e=std::max(e,std::fabs(o(j)-cur["OUT"][j])); }
    else if(kind=="PBOX"){ auto o=sando::project_point_to_box(v3("P1"),v3("P2"),cur["W"][0],cur["W"][1],cur["W"][2]); for(int j=0;j<3;j++)e=std::max(e,std::fabs(o(j)-cur["OUT"][j])); }
    else if(kind=="MT3D"){ double t=sando::min_time_double_integrator_3d(v3("P0"),v3("V0"),v3("PF"),v3("VF"),v3("VM"),v3("AM")); e=std::fabs(t-cur["T"][0]); }
    else if(kind=="CMV"){ int n=(int)cur["N"][0]; std::vector<Eigen::Vector3d> p(n); for(int i=0;i<n;i++)p[i]=Eigen::Vector3d(cur["PATH"][i*3],cur["PATH"][i*3+1],cur["PATH"][i*3+2]);
      auto o=sando::create_more_vertexes(p,cur["D"][0]); int no=(int)cur["NOUT"][0];
      if((int)o.size()!=no){e=1e9;} else for(int i=0;i<no;i++)for(int j=0;j<3;j++)e=std::max(e,std::fabs(o[i](j)-cur["OUT"][i*3+j])); }
    else if(kind=="SE3"){ Eigen::Matrix4d M; for(int i=0;i<4;i++)for(int j=0;j<4;j++)M(i,j)=cur["M"][i*4+j];
      auto Mi=sando::transform_inverse_se3(M); for(int i=0;i<4;i++)for(int j=0;j<4;j++)e=std::max(e,std::fabs(Mi(i,j)-cur["MINV"][i*4+j])); }
    upd(e); if(e>=TOL){nfail++; std::printf("  %-6s FAIL err=%.3e\n",kind.c_str(),e);}
  };

  // single-line vector tests
  auto line_test=[&](const string& key, const vector<double>& d, int stride, int outidx, auto fn){
    double e=0; for(size_t i=0;i+stride<=d.size();i+=stride){ double got=fn(d,i); e=std::max(e,std::fabs(got-d[i+outidx])); }
    upd(e); if(e>=TOL){nfail++; std::printf("  %-10s FAIL err=%.3e\n",key.c_str(),e);} else std::printf("  %-10s PASS err=%.3e\n",key.c_str(),e);
  };

  while(std::getline(f,line)){
    std::istringstream is(line);string key;is>>key;string rest;std::getline(is,rest);
    if(key=="ANGLEWRAP"){auto d=nums(rest);line_test("angle_wrap",d,2,1,[](const vector<double>&v,size_t i){return sando::angle_wrap(v[i]);});continue;}
    if(key=="ANGLEDIFF"){auto d=nums(rest);line_test("angle_diff",d,3,2,[](const vector<double>&v,size_t i){return sando::angle_diff(v[i],v[i+1]);});continue;}
    if(key=="YAWQUAT"){auto d=nums(rest); double e=0; for(size_t i=0;i+6<=d.size();i+=6){auto q=sando::quat_from_yaw(d[i]); for(int j=0;j<4;j++)e=std::max(e,std::fabs(q[j]-d[i+1+j])); e=std::max(e,std::fabs(sando::yaw_from_quat(q[0],q[1],q[2],q[3])-d[i+5]));} upd(e); std::printf("  %-10s %s err=%.3e\n","yawquat",e<TOL?"PASS":"FAIL",e); if(e>=TOL)nfail++; continue;}
    if(key=="MT1D"){auto d=nums(rest);line_test("min_time1d",d,7,6,[](const vector<double>&v,size_t i){return sando::min_time_double_integrator_1d(v[i],v[i+1],v[i+2],v[i+3],v[i+4],v[i+5]);});continue;}
    if(key=="SAT"||key=="PSPH"||key=="PBOX"||key=="MT3D"||key=="CMV"||key=="SE3"){cur.clear();kind=key;continue;}
    if(key=="END"){blk();continue;}
    cur[key]=nums(rest);
  }
  std::printf("\nutils_math golden: %d fail, global_max_err=%.3e (tol=%.0e)\n",nfail,gmax,TOL);
  std::printf("%s\n", nfail==0?"ALL PASS (C++ reproduces Python)":"FAILED");
  return nfail==0?0:1;
}
