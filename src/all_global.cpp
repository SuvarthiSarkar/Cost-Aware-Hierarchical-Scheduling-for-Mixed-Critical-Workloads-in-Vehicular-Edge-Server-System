// ALL_GLOBAL: No local mode, everything via CS. Same Alg 5 logic.
// Usage: ./all_global <input> [d_hop] [max_util] [--pidle W] [--pmax W] [--detailed] [--stats]
#include "common.h"
static double CRIT_FREQ = 0.7;
int bm(int cur,Task&t,vector<vector<double>>&u){
    double ut=(double)t.process_time/(double)(t.deadline-cur-1);
    int st=cur+1,et=t.deadline,m=(int)u.size()-1;
    if(t.flag==0&&(ut-CRIT_FREQ)<=-0.000001){int pt=(int)ceil((double)t.process_time/CRIT_FREQ);
        int ee=st+pt;double cl=0;int pr=-1;
        for(int j=1;j<=m;j++){bool ok=true;double ma=0;for(int i=st;i<=ee;i++){if(i>=(int)u[j].size()){ok=false;break;}if((u[j][i]-(MAX_UTIL-CRIT_FREQ))<=-0.000001)ma=max(ma,u[j][i]);else{ok=false;break;}}
            if(ok){if((cl-ma)<=-0.000001){pr=j;cl=ma;}else if(pr==-1){pr=j;cl=ma;}if(fabs(cl-ma)<=0.0000001&&pr!=-1)if(eu_calc_dis(t.x,t.y,j)<eu_calc_dis(t.x,t.y,pr))pr=j;}}
        if(pr!=-1){t.start_time=st;t.end_time=ee;t.utilisation=CRIT_FREQ;return pr;}}
    double val=MAX_UTIL-ut;double cl=0;int pr=-1;
    for(int j=1;j<=m;j++){bool ok=true;double ma=0;for(int i=st;i<=et;i++){if(i>=(int)u[j].size()){ok=false;break;}if((u[j][i]-val)<=-0.000001)ma=max(ma,u[j][i]);else{ok=false;break;}}
        if(ok){if((cl-ma)<=-0.000001){pr=j;cl=ma;}else if(pr==-1){pr=j;cl=ma;}if(fabs(cl-ma)<=0.0000001&&pr!=-1)if(eu_calc_dis(t.x,t.y,j)<eu_calc_dis(t.x,t.y,pr))pr=j;}}
    if(pr!=-1){t.start_time=st;t.end_time=et;t.utilisation=ut;return pr;}return -1;}
void run(vector<Task>&tasks,int m){
    int t=0,i=0;vector<Task>in;vector<vector<double>>u(m+1,vector<double>(MX+5,0.0));
    vector<vector<Task>>p(m+1);CostResult c(m); c.set_task_counts(tasks);
    while(t<=MX){sim_collect(t,i,tasks,in);sim_complete(t,m,p,c);
        if(t%BATCH==0&&!in.empty()){sort(in.begin(),in.end(),comp_priority);vector<Task>ns;
            for(auto&x:in){if(x.process_time+t+1>x.deadline){if(x.flag==0)c.drophard++;else c.dropsoft++;continue;}
                // CS overhead: all tasks go through CS
                int cs_oh=(int)ceil(SCHED_GLOBAL_MS+1.0);
                x.deadline-=cs_oh;
                if(x.process_time+t+1>x.deadline){if(x.flag==0)c.drophard++;else c.dropsoft++;continue;}
                x.utilisation=(double)x.process_time/(double)(x.deadline-t-1);
                int pr=bm(t,x,u);
                if(pr!=-1){bool run=true;if(x.flag==0)c.add_distance(x,pr);
                    else{if(d_hop_check(x.x,x.y,pr,m))c.add_distance(x,pr);else{ns.push_back(x);run=false;}}
                    if(run)assign_task(x,pr,u,p);}
                else{if(x.flag==0){set<int>dr;pr=bestprocdrop(p,x,u,t,dr);if(pr==-1)ns.push_back(x);else execute_drop(pr,x,t,p,u,dr,c);}else ns.push_back(x);}
            }in=ns;}t++;sim_energy(t,m,u,c);}
    for(auto&x:in){if(x.flag==0)c.drophard++;else c.dropsoft++;}print_stats_if_needed("ALL_GLOBAL",c);}
int main(int argc,char*argv[]){if(argc<2)return 1;check_detailed_flag(argc,argv);
    if(argc>=3&&string(argv[2])!="--detailed"&&string(argv[2])!="--stats"&&string(argv[2])!="--pidle"&&string(argv[2])!="--pmax")DIS_HOP=stoi(argv[2]);
    if(argc>=4&&string(argv[3])!="--detailed"&&string(argv[3])!="--stats"&&string(argv[3])!="--pidle"&&string(argv[3])!="--pmax")MAX_UTIL=stof(argv[3]);
    CRIT_FREQ=critical_utilization();
    vector<Task>tasks;int m=read_input(argv[1],tasks);run(tasks,m);return 0;}
