// AGA: Adaptive Genetic Algorithm. Materwala et al. (Veh.Comm. 2023)
// Optimized: vector-based eval, batch time-range clearing
#include "common.h"
static const double CR=0.95, MR=0.1;
static const int GENS=5, MAXPOP=10;

struct Ind { vector<int>a; double raw,pen,fit; };

static int g_tmin, g_tmax; // batch time range

void eval(Ind &ind, vector<Task> &b, int m, vector<vector<double>> &ub,
          int cur, vector<vector<double>> &tmp) {
    // Clear only the batch time range
    for (int j = 1; j <= m; j++)
        fill(tmp[j].begin()+g_tmin, tmp[j].begin()+g_tmax+1, 0.0);

    int n = b.size(); double tc = 0, pd = 0, pu = 0;
    for (int i = 0; i < n; i++) {
        Task &x = b[i]; int bs = ind.a[i];
        double ut = x.utilisation;
        int st = cur+1, et = min(x.deadline, g_tmax);
        bool ok = true;
        for (int t = st; t <= et; t++) {
                        if (ub[bs][t] + tmp[bs][t] + ut > bs_cap(bs) + 0.000001) { ok = false; break; }
        }
        if (ok) {
            for (int t = st; t <= et; t++) tmp[bs][t] += ut;
            tc += sqrt((double)eu_calc_dis(x.x, x.y, bs)) / (200.0*sqrt(2.0));
        } else {
            if (x.flag == 0) pd += 0.9; else pd += 0.1;
            pu += 1.0;
        }
    }
    ind.raw = tc + pd; ind.pen = pd + pu;
}

void adapt(vector<Ind> &pop) {
    int ps = pop.size(), nf = 0;
    for (auto &i : pop) if (i.pen < 0.000001) nf++;
    double g = (double)nf / ps;
    double mnc = 1e18, mxc = -1e18, mxp = -1e18;
    for (auto &i : pop) { mnc=min(mnc,i.raw); mxc=max(mxc,i.raw); mxp=max(mxp,i.pen); }
    double cr = mxc-mnc; if(cr<0.000001) cr=1; if(mxp<0.000001) mxp=1;
    for (auto &i : pop) {
        double nc=(i.raw-mnc)/cr, np=i.pen/mxp, fb;
        if(nf==0) fb=np; else if(i.pen<0.000001) fb=nc;
        else fb=sqrt(nc*nc+np*np)+(1-g)*np+g*nc;
        i.fit=1.0/(fb+1.0);
    }
}

int rsel(vector<Ind>&p,double tf){
    double r=(double)rand()/RAND_MAX*tf, c=0;
    for(int i=0;i<(int)p.size();i++){c+=p[i].fit; if(c>=r) return i;}
    return p.size()-1;
}

void run(vector<Task> &tasks, int m) {
    int t=0, i=0; vector<Task> in;
    vector<vector<double>> u(m+1, vector<double>(MX+5, 0.0));
    vector<vector<Task>> p(m+1);
    CostResult c(m); c.set_task_counts(tasks); srand(42);
    vector<vector<double>> tmp(m+1, vector<double>(MX+5, 0.0));

    while (t <= MX) {
        sim_collect(t, i, tasks, in); sim_complete(t, m, p, c);
        if (t%BATCH==0 && !in.empty()) {
            sort(in.begin(), in.end(), comp_priority);
            vector<Task> batch;
            for (auto &x : in) {
                if (x.process_time+t+1 > x.deadline) {
                    if(x.flag==0) c.drophard++; else c.dropsoft++;
                } else {
                    x.utilisation = (double)x.process_time/(double)(x.deadline-t-1);
                    batch.push_back(x);
                }
            }
            if (!batch.empty()) {
                // Compute batch time range once
                g_tmin = t+1; g_tmax = t+1;
                for (auto &x : batch) g_tmax = max(g_tmax, min(x.deadline, MX+4));

                int n=batch.size(), ps=min(2*n, MAXPOP);
                vector<Ind> pop(ps);
                for(int q=0;q<ps;q++){pop[q].a.resize(n);for(int j=0;j<n;j++)pop[q].a[j]=(rand()%m)+1;}

                for (int g=0; g<GENS; g++) {
                    for(auto &ind:pop) eval(ind,batch,m,u,t,tmp);
                    adapt(pop);
                    double tf=0; for(auto &ind:pop) tf+=ind.fit;
                    vector<Ind> sel(ps);
                    for(int q=0;q<ps;q++) sel[q]=pop[rsel(pop,tf)];
                    vector<Ind> np;
                    for(int q=0;q+1<ps;q+=2){
                        Ind &p1=sel[q],&p2=sel[q+1];
                        if((double)rand()/RAND_MAX<CR){
                            int cp=rand()%n; Ind c1,c2; c1.a.resize(n); c2.a.resize(n);
                            for(int j=0;j<n;j++){if(j<cp){c1.a[j]=p1.a[j];c2.a[j]=p2.a[j];}else{c1.a[j]=p2.a[j];c2.a[j]=p1.a[j];}}
                            eval(c1,batch,m,u,t,tmp); eval(c2,batch,m,u,t,tmp);
                            eval(p1,batch,m,u,t,tmp); eval(p2,batch,m,u,t,tmp);
                            vector<Ind*>four={&p1,&p2,&c1,&c2};
                            sort(four.begin(),four.end(),[](Ind*a,Ind*b){return a->raw<b->raw;});
                            np.push_back(*four[0]); np.push_back(*four[1]);
                        } else { np.push_back(p1); np.push_back(p2); }
                    }
                    if(ps%2==1) np.push_back(sel[ps-1]);
                    int nmut=max(1,(int)(n*ps*MR));
                    for(int mi=0;mi<nmut;mi++) np[rand()%ps].a[rand()%n]=(rand()%m)+1;
                    pop=np;
                }
                for(auto &ind:pop) eval(ind,batch,m,u,t,tmp);
                adapt(pop);
                int best=0; for(int q=1;q<ps;q++) if(pop[q].fit>pop[best].fit) best=q;
                for(int idx=0;idx<n;idx++){
                    Task &x=batch[idx]; int bs=pop[best].a[idx];
                    double ut=x.utilisation; int st=t+1, et=x.deadline;
                    if(can_fit(u,bs,ut,st,et,bs_cap(bs))){x.start_time=st;x.end_time=et;assign_task(x,bs,u,p);c.add_distance(x,bs);}
                    else{if(x.flag==0){bool f=false;for(int j=1;j<=m;j++){if(can_fit(u,j,ut,st,et,bs_cap(j))){x.start_time=st;x.end_time=et;assign_task(x,j,u,p);c.add_distance(x,j);f=true;break;}}if(!f)c.drophard++;}else c.dropsoft++;}
                }
            }
            in.clear();
        }
        t++; sim_energy(t,m,u,c);
    }
    print_stats_if_needed("AGA", c);
}

int main(int argc,char*argv[]){if(argc<2)return 1;check_detailed_flag(argc,argv);
    vector<Task>tasks;int m=read_input(argv[1],tasks);run(tasks,m);return 0;}
