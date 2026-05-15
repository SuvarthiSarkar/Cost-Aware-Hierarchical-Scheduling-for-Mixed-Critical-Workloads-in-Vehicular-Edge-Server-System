// LB: Least-loaded BS, min-freq. No dropping, no D_max, no U_max.
// Tang et al. (TVT 2020)
#include "common.h"
void run(vector<Task>&tasks,int m){
    int t=0,i=0;vector<Task>in;vector<vector<double>>u(m+1,vector<double>(MX+5,0.0));
    vector<vector<Task>>p(m+1);CostResult c(m); c.set_task_counts(tasks);
    while(t<=MX){sim_collect(t,i,tasks,in);sim_complete(t,m,p,c);
        if(t%BATCH==0&&!in.empty()){sort(in.begin(),in.end(),comp_priority);vector<Task>ns;
            for(auto&x:in){if(x.process_time+t+1>x.deadline){if(x.flag==0)c.drophard++;else c.dropsoft++;continue;}
                double ut=(double)x.process_time/(double)(x.deadline-t-1);x.utilisation=ut;
                int st=t+1,et=x.deadline,pr=-1;double ml=1e18;
                for(int j=1;j<=m;j++){if(!can_fit(u,j,ut,st,et,bs_cap(j)))continue;
                    double avg=0;int cnt=0;for(int ii=st;ii<=et;ii++){avg+=u[j][ii];cnt++;}avg/=cnt;
                    if(avg<ml-0.000001){ml=avg;pr=j;}}
                if(pr!=-1){x.start_time=st;x.end_time=et;assign_task(x,pr,u,p);c.add_distance(x,pr);}
                else ns.push_back(x);
            }in=ns;}t++;sim_energy(t,m,u,c);}
    for(auto&x:in){if(x.flag==0)c.drophard++;else c.dropsoft++;}print_stats_if_needed("LB",c);}
int main(int argc,char*argv[]){if(argc<2)return 1;check_detailed_flag(argc,argv);
    vector<Task>tasks;int m=read_input(argv[1],tasks);run(tasks,m);return 0;}
