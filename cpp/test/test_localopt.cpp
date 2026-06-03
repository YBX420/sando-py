// Golden verification for local_opt.hpp geometric core vs local_opt.py.
#include "sando_cpp/local_opt.hpp"
#include "sando_cpp/obstacles.hpp"
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
  const char* path = (argc>1)?argv[1]:"golden/localopt_cases.txt";
  std::ifstream f(path); if(!f){std::printf("CANNOT OPEN %s\n",path);return 2;}
  std::map<string,vector<double>> cur; string kind, line;
  int n=0,nfail=0; double gmax=0; const double TOL=1e-9;
  auto v3=[&](const char* k){return Eigen::Vector3d(cur[k][0],cur[k][1],cur[k][2]);};

  auto process=[&](){
    double e=0;
    if(kind=="SDG_SPH"){
      sando::SphereObstacle s(v3("C0"),cur["R"][0],v3("V"),"human",v3("A"));
      auto g=sando::signed_dist_and_grad(&s, v3("P"), cur["T"][0]);
      e=std::max(e,std::fabs(g.d-cur["D"][0]));
      for(int j=0;j<3;j++) e=std::max(e,std::fabs(g.dd_dp(j)-cur["DDP"][j]));
      e=std::max(e,std::fabs(g.dd_dt-cur["DDT"][0]));
    } else if(kind=="SDG_BOX"){
      sando::AABBObstacle b(v3("LO"),v3("HI"),"wall");
      auto g=sando::signed_dist_and_grad(&b, v3("P"), cur["T"][0]);
      e=std::max(e,std::fabs(g.d-cur["D"][0]));
      for(int j=0;j<3;j++) e=std::max(e,std::fabs(g.dd_dp(j)-cur["DDP"][j]));
      e=std::max(e,std::fabs(g.dd_dt-cur["DDT"][0]));
    } else if(kind=="VANDER"){
      int k=(int)cur["K"][0];
      Eigen::VectorXd taus=Eigen::Map<Eigen::VectorXd>(cur["TAUS"].data(),k);
      auto B=sando::seg_vander(taus);
      const char* bk[4]={"B0","B1","B2","B3"};
      for(int o=0;o<4;o++) for(int i=0;i<k;i++) for(int p=0;p<6;p++)
        e=std::max(e,std::fabs(B[o](i,p)-cur[bk[o]][i*6+p]));
    }
    gmax=std::max(gmax,e); bool ok=e<TOL;
    std::printf("  %-8s : max_err=%.3e  %s\n",kind.c_str(),e,ok?"PASS":"FAIL");
    if(!ok)nfail++; n++;
  };

  while(std::getline(f,line)){
    if(line=="SDG_SPH"||line=="SDG_BOX"||line=="VANDER"){cur.clear();kind=line;continue;}
    if(line=="END"){process();continue;}
    std::istringstream is(line);string key;is>>key;string rest;std::getline(is,rest);cur[key]=nums(rest);
  }
  std::printf("\nlocal_opt geom golden: %d cases, %d fail, global_max_err=%.3e (tol=%.0e)\n",n,nfail,gmax,TOL);
  std::printf("%s\n", nfail==0?"ALL PASS (C++ reproduces Python)":"FAILED");
  return nfail==0?0:1;
}
