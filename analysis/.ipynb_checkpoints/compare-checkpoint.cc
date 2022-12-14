
void compare(){
  gStyle->SetOptStat(kFALSE);
  int nMaxEvt =9371;
  //int nMaxEvt =10000;

  //TFile* f1=new TFile("/afs/cern.ch/user/k/keli/eos/Faser/alignment/misalign_MC/iter20_x_y_rz_ift_unbiased_fix3layers/kfalignment_mc_iter19.root");
  TFile* f1=new TFile("../fluka_mc_test.root");
  TTree* t1=(TTree*)f1->Get("trackParam");

  //TFile* f2=new TFile("/eos/user/k/keli/Faser/alignment/data/8023_8025_8115_x_y_rz_ift_biased/kfalignment_data_iter0.root");
  TFile* f2=new TFile("../kfalignment_data_iter0.root");
  //TFile* f2=new TFile("/afs/cern.ch/user/k/keli/eos/Faser/alignment/misalign_MC/iter20_x_y_rz_ift_unbiased_fix3layers/kfalignment_mc_iter0.root");
  //TFile* f2=new TFile("/afs/cern.ch/user/k/keli/eos/Faser/alignment/misalign_MC/iter20_x_y_rz_ift_unbiased_fix3layers/kfalignment_mc_iter19.root");
  TTree* t2=(TTree*)f2->Get("trackParam");

  // TFile* f3=new TFile("/eos/user/k/keli/Faser/alignment/data/8023_8025_8115_x_y_rz_ift_biased/kfalignment_data_iter0.root");
  // TTree* t3=(TTree*)f3->Get("trackParam");

  //TString outputname="ideal_geo_MC_vs_aligned_MC";
  //TString outputname="ideal_geo_MC_vs_mis_aligned_MC";
  // TString outputname="ideal_geo_MC_vs_data";
  TString outputname="ideal_geo_MC_vs_aligned_data";


  std::vector<double>* m_fitParam_x=0;
  std::vector<double>* m_fitParam_y=0;
  std::vector<double>* m_fitParam_z=0;
  std::vector<double>* m_fitParam_pull_x=0;
  std::vector<double>* m_fitParam_pull_y=0;
  std::vector<double>* m_fitParam_chi2=0;
  std::vector<double>* m_fitParam_pz=0;
  std::vector<double>* m_fitParam_py=0;
  std::vector<double>* m_fitParam_px=0;
  std::vector<std::vector<double>>* m_fitParam_align_local_residual_x_sp=0;
  std::vector<std::vector<double>>* m_fitParam_align_local_pull_x_sp=0;
  std::vector<std::vector<double>>* m_fitParam_align_local_measured_x_sp=0;
  std::vector<std::vector<double>>* m_fitParam_align_local_measured_xe_sp=0;
  std::vector<std::vector<double>>* m_fitParam_align_local_fitted_x_sp=0;
  std::vector<std::vector<double>>* m_fitParam_align_stationId_sp=0;
  std::vector<std::vector<double>>* m_fitParam_align_layerId_sp=0;
  std::vector<std::vector<double>>* m_fitParam_align_moduleId_sp=0;
  std::vector<std::vector<double>>* m_fitParam_align_stereoId=0;
  t1->SetBranchAddress("fitParam_x",&m_fitParam_x);
  t1->SetBranchAddress("fitParam_y",&m_fitParam_y);
  t1->SetBranchAddress("fitParam_z",&m_fitParam_z);
  t1->SetBranchAddress("fitParam_chi2",&m_fitParam_chi2);
  t1->SetBranchAddress("fitParam_pz",&m_fitParam_pz);
  t1->SetBranchAddress("fitParam_py",&m_fitParam_py);
  t1->SetBranchAddress("fitParam_px",&m_fitParam_px);
  t1->SetBranchAddress("fitParam_align_stereoId",&m_fitParam_align_stereoId);
  t1->SetBranchAddress("fitParam_align_local_residual_x_sp",&m_fitParam_align_local_residual_x_sp);
  t1->SetBranchAddress("fitParam_align_local_pull_x_sp",&m_fitParam_align_local_pull_x_sp);
  t1->SetBranchAddress("fitParam_align_local_measured_x_sp",&m_fitParam_align_local_measured_x_sp);
  t1->SetBranchAddress("fitParam_align_local_measured_xe_sp",&m_fitParam_align_local_measured_xe_sp);
  t1->SetBranchAddress("fitParam_align_local_fitted_x_sp",&m_fitParam_align_local_fitted_x_sp);
  t1->SetBranchAddress("fitParam_align_layerId_sp",&m_fitParam_align_layerId_sp);
  t1->SetBranchAddress("fitParam_align_moduleId_sp",&m_fitParam_align_moduleId_sp);
  t1->SetBranchAddress("fitParam_align_stationId_sp",&m_fitParam_align_stationId_sp);

  t2->SetBranchAddress("fitParam_x",&m_fitParam_x);
  t2->SetBranchAddress("fitParam_y",&m_fitParam_y);
  t2->SetBranchAddress("fitParam_z",&m_fitParam_z);
  t2->SetBranchAddress("fitParam_chi2",&m_fitParam_chi2);
  t2->SetBranchAddress("fitParam_pz",&m_fitParam_pz);
  t2->SetBranchAddress("fitParam_py",&m_fitParam_py);
  t2->SetBranchAddress("fitParam_px",&m_fitParam_px);
  t2->SetBranchAddress("fitParam_align_stereoId",&m_fitParam_align_stereoId);
  t2->SetBranchAddress("fitParam_align_local_residual_x_sp",&m_fitParam_align_local_residual_x_sp);
  t2->SetBranchAddress("fitParam_align_local_pull_x_sp",&m_fitParam_align_local_pull_x_sp);
  t2->SetBranchAddress("fitParam_align_local_measured_x_sp",&m_fitParam_align_local_measured_x_sp);
  t2->SetBranchAddress("fitParam_align_local_measured_xe_sp",&m_fitParam_align_local_measured_xe_sp);
  t2->SetBranchAddress("fitParam_align_local_fitted_x_sp",&m_fitParam_align_local_fitted_x_sp);
  t2->SetBranchAddress("fitParam_align_layerId_sp",&m_fitParam_align_layerId_sp);
  t2->SetBranchAddress("fitParam_align_moduleId_sp",&m_fitParam_align_moduleId_sp);
  t2->SetBranchAddress("fitParam_align_stationId_sp",&m_fitParam_align_stationId_sp);

  // t3->SetBranchAddress("fitParam_x",&m_fitParam_x);
  // t3->SetBranchAddress("fitParam_y",&m_fitParam_y);
  // t3->SetBranchAddress("fitParam_chi2",&m_fitParam_chi2);
  // t3->SetBranchAddress("fitParam_pz",&m_fitParam_pz);
  // t3->SetBranchAddress("fitParam_py",&m_fitParam_py);
  // t3->SetBranchAddress("fitParam_px",&m_fitParam_px);
  // t3->SetBranchAddress("fitParam_align_stereoId",&m_fitParam_align_stereoId);
  // t3->SetBranchAddress("fitParam_align_local_residual_x_sp",&m_fitParam_align_local_residual_x_sp);
  // t3->SetBranchAddress("fitParam_align_local_pull_x_sp",&m_fitParam_align_local_pull_x_sp);
  // t3->SetBranchAddress("fitParam_align_local_measured_x_sp",&m_fitParam_align_local_measured_x_sp);
  // t3->SetBranchAddress("fitParam_align_local_measured_xe_sp",&m_fitParam_align_local_measured_xe_sp);
  // t3->SetBranchAddress("fitParam_align_local_fitted_x_sp",&m_fitParam_align_local_fitted_x_sp);
  // t3->SetBranchAddress("fitParam_align_layerId_sp",&m_fitParam_align_layerId_sp);
  // t3->SetBranchAddress("fitParam_align_moduleId_sp",&m_fitParam_align_moduleId_sp);
  // t3->SetBranchAddress("fitParam_align_stationId_sp",&m_fitParam_align_stationId_sp);

  TH1F* h1_residual_x=new TH1F("h1_residaul_x","",100,-0.5,0.5);
  TH1F* h1_pull_x=new TH1F("h1_pull_x","",100,-5,5);
  TH1F* h1_fit_chi2=new TH1F("h1_fit_chi2","",100,0,100);
  TH1F* h1_residual_x_sta0=new TH1F("h1_residaul_x_sta0","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta0_layer0=new TH1F("h1_residaul_x_sta0_layer0","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta0_layer1=new TH1F("h1_residaul_x_sta0_layer1","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta0_layer2=new TH1F("h1_residaul_x_sta0_layer2","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta1=new TH1F("h1_residaul_x_sta1","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta1_layer0=new TH1F("h1_residaul_x_sta1_layer0","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta1_layer1=new TH1F("h1_residaul_x_sta1_layer1","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta1_layer2=new TH1F("h1_residaul_x_sta1_layer2","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta2=new TH1F("h1_residaul_x_sta2","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta2_layer0=new TH1F("h1_residaul_x_sta2_layer0","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta2_layer1=new TH1F("h1_residaul_x_sta2_layer1","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta2_layer2=new TH1F("h1_residaul_x_sta2_layer2","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta3=new TH1F("h1_residaul_x_sta3","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta3_layer0=new TH1F("h1_residaul_x_sta3_layer0","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta3_layer1=new TH1F("h1_residaul_x_sta3_layer1","",100,-0.2,0.2);
  TH1F* h1_residual_x_sta3_layer2=new TH1F("h1_residaul_x_sta3_layer2","",100,-0.2,0.2);
  TH1F* h1_pz=new TH1F("h1_pz","",100,350,1350);
  TH1F* h1_py=new TH1F("h1_py","",100,0,100);
  TH1F* h1_px=new TH1F("h1_px","",100,0,100);
  TH1F* h1_x=new TH1F("h1_x","",100,-200,200);
  TH1F* h1_y=new TH1F("h1_y","",100,-200,200);
  TH1F* h1_z=new TH1F("h1_z","",100,300,2500);
  TH1F* h1_tx=new TH1F("h1_tx","",100,-0.1,0.1);
  TH1F* h1_ty=new TH1F("h1_ty","",100,-0.1,0.1);

  TH1F* h2_residual_x=new TH1F("h2_residaul_x","",100,-0.5,0.5);
  TH1F* h2_pull_x=new TH1F("h2_pull_x","",100,-5,5);
  TH1F* h2_fit_chi2=new TH1F("h2_fit_chi2","",100,0,100);
  TH1F* h2_residual_x_sta0=new TH1F("h2_residaul_x_sta0","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta1=new TH1F("h2_residaul_x_sta1","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta2=new TH1F("h2_residaul_x_sta2","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta3=new TH1F("h2_residaul_x_sta3","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta0_layer0=new TH1F("h2_residaul_x_sta0_layer0","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta0_layer1=new TH1F("h2_residaul_x_sta0_layer1","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta0_layer2=new TH1F("h2_residaul_x_sta0_layer2","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta1_layer0=new TH1F("h2_residaul_x_sta1_layer0","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta1_layer1=new TH1F("h2_residaul_x_sta1_layer1","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta1_layer2=new TH1F("h2_residaul_x_sta1_layer2","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta2_layer0=new TH1F("h2_residaul_x_sta2_layer0","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta2_layer1=new TH1F("h2_residaul_x_sta2_layer1","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta2_layer2=new TH1F("h2_residaul_x_sta2_layer2","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta3_layer0=new TH1F("h2_residaul_x_sta3_layer0","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta3_layer1=new TH1F("h2_residaul_x_sta3_layer1","",100,-0.2,0.2);
  TH1F* h2_residual_x_sta3_layer2=new TH1F("h2_residaul_x_sta3_layer2","",100,-0.2,0.2);
  TH1F* h2_pz=new TH1F("h2_pz","",100,300,1350);
  TH1F* h2_py=new TH1F("h2_py","",100,0,100);
  TH1F* h2_px=new TH1F("h2_px","",100,0,100);
  TH1F* h2_x= new TH1F("h2_x","",100,-200,200);
  TH1F* h2_y= new TH1F("h2_y","",100,-200,200);
  TH1F* h2_z= new TH1F("h2_z","",100,300,2500);
  TH1F* h2_tx=new TH1F("h2_tx","",100,-0.1,0.1);
  TH1F* h2_ty=new TH1F("h2_ty","",100,-0.1,0.1);

  // TH1F* h3_residual_x=new TH1F("h3_residaul_x","",100,-0.5,0.5);
  // TH1F* h3_pull_x=new TH1F("h3_pull_x","",100,-5,5);
  // TH1F* h3_fit_chi2=new TH1F("h3_fit_chi2","",100,0,100);
  // TH1F* h3_residual_x_sta0=new TH1F("h3_residaul_x_sta0","",100,-0.2,0.2);
  // TH1F* h3_residual_x_sta1=new TH1F("h3_residaul_x_sta1","",100,-0.2,0.2);
  // TH1F* h3_residual_x_sta2=new TH1F("h3_residaul_x_sta2","",100,-0.2,0.2);
  // TH1F* h3_residual_x_sta3=new TH1F("h3_residaul_x_sta3","",100,-0.2,0.2);
  // TH1F* h3_pz=new TH1F("h3_pz","",100,300,1350);

  //loop over all the entries
  for(int ievt=0;ievt<nMaxEvt;ievt++){
  //for(int ievt=0;ievt<t1->GetEntries();ievt++){
    t1->GetEntry(ievt);
    // if(ievt%10000==0)std::cout<<"processing "<<ievt<<" event"<<std::endl;
    //std::cout<<"TTTTTTTT: " << m_fitParam_align_local_residual_x_sp->size() << ":::::: " << m_fitParam_align_local_residual_x_sp->at(0).size() << std::endl;
    //std::cout<<"PPPPPPP: " << m_fitParam_x->size() << ":::::: " <<  m_fitParam_x->at(0) << std::endl;

    if (m_fitParam_align_local_residual_x_sp->size()==0) continue;

    for(int i=0;i<1;i++){

    //for(int i=0;i<m_fitParam_x->size();i++){

      if(m_fitParam_chi2->at(i)>100||m_fitParam_pz->at(i)<300) continue;
//    }
//    for(int i=0;i<m_fitParam_align_local_residual_x_sp->size();i+=1){
      if(m_fitParam_align_local_residual_x_sp->at(i).size()<15)continue;//only use good track

      h1_fit_chi2->Fill(m_fitParam_chi2->at(i));
      h1_pz->Fill(m_fitParam_pz->at(i));
      h1_py->Fill(m_fitParam_py->at(i));
      h1_px->Fill(m_fitParam_px->at(i));
      h1_z->Fill(sqrt(pow(m_fitParam_pz->at(i),2)+pow(m_fitParam_py->at(i),2)+pow(m_fitParam_px->at(i),2)));
      h1_y->Fill(m_fitParam_y->at(i));
      h1_x->Fill(m_fitParam_x->at(i));
      h1_tx->Fill(m_fitParam_px->at(i)/m_fitParam_pz->at(i));
      h1_ty->Fill(m_fitParam_py->at(i)/m_fitParam_pz->at(i));

      int offset=0;
      for(int j=0;j<m_fitParam_align_local_residual_x_sp->at(i).size();j++){
	      h1_residual_x->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
	      h1_pull_x->Fill(m_fitParam_align_local_pull_x_sp->at(i).at(j));
        // per station residual
	      if(m_fitParam_align_stationId_sp->at(i).at(j)==0){
	        h1_residual_x_sta0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          //per later residual 
          if(m_fitParam_align_layerId_sp->at(i).at(j)==0){
            h1_residual_x_sta0_layer0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          }
          if(m_fitParam_align_layerId_sp->at(i).at(j)==1){
            h1_residual_x_sta0_layer1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          }
          if(m_fitParam_align_layerId_sp->at(i).at(j)==2){
            h1_residual_x_sta0_layer2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          }
        }
	      if(m_fitParam_align_stationId_sp->at(i).at(j)==1){
	        h1_residual_x_sta1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          //per later residual 
          if(m_fitParam_align_layerId_sp->at(i).at(j)==0)
            h1_residual_x_sta1_layer0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==1)
            h1_residual_x_sta1_layer1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==2)
            h1_residual_x_sta1_layer2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
        }
	      if(m_fitParam_align_stationId_sp->at(i).at(j)==2){
	        h1_residual_x_sta2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          //per later residual 
          if(m_fitParam_align_layerId_sp->at(i).at(j)==0)
            h1_residual_x_sta2_layer0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==1)
            h1_residual_x_sta2_layer1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==2)
            h1_residual_x_sta2_layer2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
        }
	      if(m_fitParam_align_stationId_sp->at(i).at(j)==3){
	        h1_residual_x_sta3->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          //per later residual 
          if(m_fitParam_align_layerId_sp->at(i).at(j)==0)
            h1_residual_x_sta3_layer0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==1)
            h1_residual_x_sta3_layer1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==2)
            h1_residual_x_sta3_layer2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
        }

      }
    }
  }

  for(int ievt=0;ievt<nMaxEvt;ievt++){
    //for(int ievt=0;ievt<t2->GetEntries();ievt++){
    t2->GetEntry(ievt);
    // if(ievt%10000==0)std::cout<<"processing "<<ievt<<" event"<<std::endl;
    if (m_fitParam_align_local_residual_x_sp->size()==0) continue;

    for(int i=0;i<1;i++){
    //for(int i=0;i<m_fitParam_x->size();i++){
      //if(m_fitParam_chi2->at(i)>200) continue;
      if(m_fitParam_chi2->at(i)>100||m_fitParam_pz->at(i)<300) continue;
//    }
//    for(int i=0;i<m_fitParam_align_local_residual_x_sp->size();i+=1){
      if(m_fitParam_align_local_residual_x_sp->at(i).size()<15)continue;//only use good track

      h2_fit_chi2->Fill(m_fitParam_chi2->at(i));
      h2_pz->Fill(m_fitParam_pz->at(i));
      h2_py->Fill(m_fitParam_py->at(i));
      h2_px->Fill(m_fitParam_px->at(i));
      h2_z->Fill(sqrt(pow(m_fitParam_pz->at(i),2)+pow(m_fitParam_py->at(i),2)+pow(m_fitParam_px->at(i),2)));
      h2_y->Fill(m_fitParam_y->at(i));
      h2_x->Fill(m_fitParam_x->at(i));
      h2_tx->Fill(m_fitParam_px->at(i)/m_fitParam_pz->at(i));
      h2_ty->Fill(m_fitParam_py->at(i)/m_fitParam_pz->at(i));
      int offset=0;
      for(int j=0;j<m_fitParam_align_local_residual_x_sp->at(i).size();j++){
	      h2_residual_x->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
	      h2_pull_x->Fill(m_fitParam_align_local_pull_x_sp->at(i).at(j));
	      if(m_fitParam_align_stationId_sp->at(i).at(j)==0){
	        h2_residual_x_sta0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          //per later residual 
          if(m_fitParam_align_layerId_sp->at(i).at(j)==0){
            h2_residual_x_sta0_layer0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          }
          if(m_fitParam_align_layerId_sp->at(i).at(j)==1){
            h2_residual_x_sta0_layer1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          }
          if(m_fitParam_align_layerId_sp->at(i).at(j)==2){
            h2_residual_x_sta0_layer2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          }
        }
	      if(m_fitParam_align_stationId_sp->at(i).at(j)==1){
	        h2_residual_x_sta1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          //per later residual 
          if(m_fitParam_align_layerId_sp->at(i).at(j)==0)
            h2_residual_x_sta1_layer0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==1)
            h2_residual_x_sta1_layer1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==2)
            h2_residual_x_sta1_layer2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
        }
	      if(m_fitParam_align_stationId_sp->at(i).at(j)==2){
	        h2_residual_x_sta2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          //per later residual 
          if(m_fitParam_align_layerId_sp->at(i).at(j)==0)
            h2_residual_x_sta2_layer0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==1)
            h2_residual_x_sta2_layer1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==2)
            h2_residual_x_sta2_layer2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
        }
	      if(m_fitParam_align_stationId_sp->at(i).at(j)==3){
	        h2_residual_x_sta3->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          //per later residual 
          if(m_fitParam_align_layerId_sp->at(i).at(j)==0)
            h2_residual_x_sta3_layer0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==1)
            h2_residual_x_sta3_layer1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
          if(m_fitParam_align_layerId_sp->at(i).at(j)==2)
            h2_residual_x_sta3_layer2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
        }
      }
    }
  }



//   for(int ievt=0;ievt<nMaxEvt;ievt++){
//     //for(int ievt=0;ievt<t3->GetEntries();ievt++){
//     t3->GetEntry(ievt);
//     if(ievt%10000==0)std::cout<<"processing "<<ievt<<" event"<<std::endl;
//     for(int i=0;i<1;i++){
//       //if(m_fitParam_chi2->at(i)>200) continue;
//       if(m_fitParam_chi2->at(i)>100||m_fitParam_pz->at(i)<300) continue;
// //    }
// //    for(int i=0;i<m_fitParam_align_local_residual_x_sp->size();i+=1){
//       if(m_fitParam_align_local_residual_x_sp->at(i).size()<15)continue;//only use good track
//       h3_fit_chi2->Fill(m_fitParam_chi2->at(i));
//       h3_pz->Fill(m_fitParam_pz->at(i));
//       int offset=0;
//       for(int j=0;j<m_fitParam_align_local_residual_x_sp->at(i).size();j++){
// 	h3_residual_x->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
// 	h3_pull_x->Fill(m_fitParam_align_local_pull_x_sp->at(i).at(j));
// 	if(m_fitParam_align_stationId_sp->at(i).at(j)==0)
// 	  h3_residual_x_sta0->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
// 	if(m_fitParam_align_stationId_sp->at(i).at(j)==1)
// 	  //if(m_fitParam_align_stationId_sp->at(i).at(j)==1&&m_fitParam_align_layerId_sp->at(i).at(j)==0)
// 	  h3_residual_x_sta1->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
// 	if(m_fitParam_align_stationId_sp->at(i).at(j)==2)
// 	  h3_residual_x_sta2->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
// 	if(m_fitParam_align_stationId_sp->at(i).at(j)==3)
// 	  h3_residual_x_sta3->Fill(m_fitParam_align_local_residual_x_sp->at(i).at(j));
//       }
//     }
//   }
  std::cout<<"like finish reading"<<std::endl;
  //fitting functions
  TF1* f1_gaus=new TF1("f1_gaus","gaus(0)",-3,3);
  TF1* f1_2gaus=new TF1("f1_2gaus","gaus(0)+gaus(3)",-0.3,0.3);
  TF1* f1_landau=new TF1("f1_landau","landau(0)",0,20);
  TLatex tex; tex.SetNDC();
  tex.SetTextSize(0.035);

  //comparison
  TCanvas* c_res_com = new TCanvas("c_res_com");
  TLegend * leg=new TLegend(.55,.65,.9,.9);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  h1_residual_x->Draw();
  h1_residual_x->GetXaxis()->SetTitle("residual_x (mm)");
  h1_residual_x->SetLineColor(94);
  h1_residual_x->SetFillColor(94);
  h1_residual_x->SetLineWidth(3);
  h2_residual_x->SetLineWidth(3);
  // h3_residual_x->SetLineWidth(3);
  h2_residual_x->SetLineColor(kBlack);
  // h3_residual_x->SetLineColor(kRed);
  h2_residual_x->Draw("same");
  // h3_residual_x->Draw("same");
  h2_residual_x->Scale(h1_residual_x->Integral()/h2_residual_x->Integral());
  // h3_residual_x->Scale(h1_residual_x->Integral()/h3_residual_x->Integral());
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  leg->AddEntry(h1_residual_x, "Ideal Geometry MC", "L");
  leg->AddEntry(h2_residual_x, "Aligned Data", "L");
  //leg->AddEntry(h2_residual_x, "Mis-aligned MC", "L");
  //leg->AddEntry(h2_residual_x, "Aligned MC", "L");


  // leg->AddEntry(h3_residual_x, "mis-aligned MC w/ alignment", "L");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  // f1_gaus->SetLineColor(kRed);
  // h3_residual_x->Fit(f1_gaus,"L","",-0.05,0.05);
  // tex.SetTextColor(kRed);
  // tex.DrawLatex(0.15,0.55,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  // tex.DrawLatex(0.15,0.50,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com->SaveAs("../docs/"+outputname+"_com_res.png");


  TCanvas* c_res_com_sta0 = new TCanvas("c_res_com_sta0");
  h1_residual_x_sta0->Draw();
  h1_residual_x_sta0->GetXaxis()->SetTitle("residual_x_station0 (mm)");
  h1_residual_x_sta0->SetLineColor(94);
  h1_residual_x_sta0->SetFillColor(94);
  h1_residual_x_sta0->SetLineWidth(3);
  h2_residual_x_sta0->SetLineWidth(3);
  // h3_residual_x_sta0->SetLineWidth(3);
  h2_residual_x_sta0->SetLineColor(kBlack);
  // h3_residual_x_sta0->SetLineColor(kRed);
  h2_residual_x_sta0->Scale(h1_residual_x_sta0->Integral()/h2_residual_x_sta0->Integral());
  // h3_residual_x_sta0->Scale(h1_residual_x_sta0->Integral()/h3_residual_x_sta0->Integral());
  h2_residual_x_sta0->Draw("same");
  // h3_residual_x_sta0->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  // f1_gaus->SetLineColor(kRed);
  // h3_residual_x_sta0->Fit(f1_gaus,"L","",-0.05,0.05);
  // tex.SetTextColor(kRed);
  // tex.DrawLatex(0.15,0.55,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  // tex.DrawLatex(0.15,0.50,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta0->SaveAs("../docs/"+outputname+"_com_res_sta0.png");

  TCanvas* c_res_com_sta0_layer0 = new TCanvas("c_res_com_sta0_layer0");
  h1_residual_x_sta0_layer0->Draw();
  h1_residual_x_sta0_layer0->GetXaxis()->SetTitle("residual_x_station0_layer0 (mm)");
  h1_residual_x_sta0_layer0->SetLineColor(94);
  h1_residual_x_sta0_layer0->SetFillColor(94);
  h1_residual_x_sta0_layer0->SetLineWidth(3);
  h2_residual_x_sta0_layer0->SetLineWidth(3);
  h2_residual_x_sta0_layer0->SetLineColor(kBlack);
  h2_residual_x_sta0_layer0->Scale(h1_residual_x_sta0_layer0->Integral()/h2_residual_x_sta0_layer0->Integral());
  h2_residual_x_sta0_layer0->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta0_layer0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta0_layer0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta0_layer0->SaveAs("../docs/"+outputname+"_com_res_sta0_layer0.png");

  TCanvas* c_res_com_sta0_layer1 = new TCanvas("c_res_com_sta0_layer1");
  h1_residual_x_sta0_layer1->Draw();
  h1_residual_x_sta0_layer1->GetXaxis()->SetTitle("residual_x_station0_layer1 (mm)");
  h1_residual_x_sta0_layer1->SetLineColor(94);
  h1_residual_x_sta0_layer1->SetFillColor(94);
  h1_residual_x_sta0_layer1->SetLineWidth(3);
  h2_residual_x_sta0_layer1->SetLineWidth(3);
  h2_residual_x_sta0_layer1->SetLineColor(kBlack);
  h2_residual_x_sta0_layer1->Scale(h1_residual_x_sta0_layer1->Integral()/h2_residual_x_sta0_layer1->Integral());
  h2_residual_x_sta0_layer1->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta0_layer1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta0_layer1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta0_layer1->SaveAs("../docs/"+outputname+"_com_res_sta0_layer1.png");

  TCanvas* c_res_com_sta0_layer2 = new TCanvas("c_res_com_sta0_layer2");
  h1_residual_x_sta0_layer2->Draw();
  h1_residual_x_sta0_layer2->GetXaxis()->SetTitle("residual_x_station0_layer2 (mm)");
  h1_residual_x_sta0_layer2->SetLineColor(94);
  h1_residual_x_sta0_layer2->SetFillColor(94);
  h1_residual_x_sta0_layer2->SetLineWidth(3);
  h2_residual_x_sta0_layer2->SetLineWidth(3);
  h2_residual_x_sta0_layer2->SetLineColor(kBlack);
  h2_residual_x_sta0_layer2->Scale(h1_residual_x_sta0_layer2->Integral()/h2_residual_x_sta0_layer2->Integral());
  h2_residual_x_sta0_layer2->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta0_layer2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta0_layer2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta0_layer2->SaveAs("../docs/"+outputname+"_com_res_sta0_layer2.png");




  TCanvas* c_res_com_sta1_layer0 = new TCanvas("c_res_com_sta1_layer0");
  h1_residual_x_sta1_layer0->Draw();
  h1_residual_x_sta1_layer0->GetXaxis()->SetTitle("residual_x_station1_layer0 (mm)");
  h1_residual_x_sta1_layer0->SetLineColor(94);
  h1_residual_x_sta1_layer0->SetFillColor(94);
  h1_residual_x_sta1_layer0->SetLineWidth(3);
  h2_residual_x_sta1_layer0->SetLineWidth(3);
  h2_residual_x_sta1_layer0->SetLineColor(kBlack);
  h2_residual_x_sta1_layer0->Scale(h1_residual_x_sta1_layer0->Integral()/h2_residual_x_sta1_layer0->Integral());
  h2_residual_x_sta1_layer0->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta1_layer0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta1_layer0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta1_layer0->SaveAs("../docs/"+outputname+"_com_res_sta1_layer0.png");

  TCanvas* c_res_com_sta1_layer1 = new TCanvas("c_res_com_sta1_layer1");
  h1_residual_x_sta1_layer1->Draw();
  h1_residual_x_sta1_layer1->GetXaxis()->SetTitle("residual_x_station1_layer1 (mm)");
  h1_residual_x_sta1_layer1->SetLineColor(94);
  h1_residual_x_sta1_layer1->SetFillColor(94);
  h1_residual_x_sta1_layer1->SetLineWidth(3);
  h2_residual_x_sta1_layer1->SetLineWidth(3);
  h2_residual_x_sta1_layer1->SetLineColor(kBlack);
  h2_residual_x_sta1_layer1->Scale(h1_residual_x_sta1_layer1->Integral()/h2_residual_x_sta1_layer1->Integral());
  h2_residual_x_sta1_layer1->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta1_layer1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta1_layer1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta1_layer1->SaveAs("../docs/"+outputname+"_com_res_sta1_layer1.png");

  TCanvas* c_res_com_sta1_layer2 = new TCanvas("c_res_com_sta1_layer2");
  h1_residual_x_sta1_layer2->Draw();
  h1_residual_x_sta1_layer2->GetXaxis()->SetTitle("residual_x_station1_layer2 (mm)");
  h1_residual_x_sta1_layer2->SetLineColor(94);
  h1_residual_x_sta1_layer2->SetFillColor(94);
  h1_residual_x_sta1_layer2->SetLineWidth(3);
  h2_residual_x_sta1_layer2->SetLineWidth(3);
  h2_residual_x_sta1_layer2->SetLineColor(kBlack);
  h2_residual_x_sta1_layer2->Scale(h1_residual_x_sta1_layer2->Integral()/h2_residual_x_sta1_layer2->Integral());
  h2_residual_x_sta1_layer2->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta1_layer2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta1_layer2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta1_layer2->SaveAs("../docs/"+outputname+"_com_res_sta1_layer2.png");


  TCanvas* c_res_com_sta2_layer0 = new TCanvas("c_res_com_sta2_layer0");
  h1_residual_x_sta2_layer0->Draw();
  h1_residual_x_sta2_layer0->GetXaxis()->SetTitle("residual_x_station2_layer0 (mm)");
  h1_residual_x_sta2_layer0->SetLineColor(94);
  h1_residual_x_sta2_layer0->SetFillColor(94);
  h1_residual_x_sta2_layer0->SetLineWidth(3);
  h2_residual_x_sta2_layer0->SetLineWidth(3);
  h2_residual_x_sta2_layer0->SetLineColor(kBlack);
  h2_residual_x_sta2_layer0->Scale(h1_residual_x_sta2_layer0->Integral()/h2_residual_x_sta2_layer0->Integral());
  h2_residual_x_sta2_layer0->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta2_layer0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta2_layer0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta2_layer0->SaveAs("../docs/"+outputname+"_com_res_sta2_layer0.png");

  TCanvas* c_res_com_sta2_layer1 = new TCanvas("c_res_com_sta2_layer1");
  h1_residual_x_sta2_layer1->Draw();
  h1_residual_x_sta2_layer1->GetXaxis()->SetTitle("residual_x_station2_layer1 (mm)");
  h1_residual_x_sta2_layer1->SetLineColor(94);
  h1_residual_x_sta2_layer1->SetFillColor(94);
  h1_residual_x_sta2_layer1->SetLineWidth(3);
  h2_residual_x_sta2_layer1->SetLineWidth(3);
  h2_residual_x_sta2_layer1->SetLineColor(kBlack);
  h2_residual_x_sta2_layer1->Scale(h1_residual_x_sta2_layer1->Integral()/h2_residual_x_sta2_layer1->Integral());
  h2_residual_x_sta2_layer1->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta2_layer1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta2_layer1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta2_layer1->SaveAs("../docs/"+outputname+"_com_res_sta2_layer1.png");

  TCanvas* c_res_com_sta2_layer2 = new TCanvas("c_res_com_sta2_layer2");
  h1_residual_x_sta2_layer2->Draw();
  h1_residual_x_sta2_layer2->GetXaxis()->SetTitle("residual_x_station2_layer2 (mm)");
  h1_residual_x_sta2_layer2->SetLineColor(94);
  h1_residual_x_sta2_layer2->SetFillColor(94);
  h1_residual_x_sta2_layer2->SetLineWidth(3);
  h2_residual_x_sta2_layer2->SetLineWidth(3);
  h2_residual_x_sta2_layer2->SetLineColor(kBlack);
  h2_residual_x_sta2_layer2->Scale(h1_residual_x_sta2_layer2->Integral()/h2_residual_x_sta2_layer2->Integral());
  h2_residual_x_sta2_layer2->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta2_layer2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta2_layer2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta2_layer2->SaveAs("../docs/"+outputname+"_com_res_sta2_layer2.png");



  TCanvas* c_res_com_sta3_layer0 = new TCanvas("c_res_com_sta3_layer0");
  h1_residual_x_sta3_layer0->Draw();
  h1_residual_x_sta3_layer0->GetXaxis()->SetTitle("residual_x_station3_layer0 (mm)");
  h1_residual_x_sta3_layer0->SetLineColor(94);
  h1_residual_x_sta3_layer0->SetFillColor(94);
  h1_residual_x_sta3_layer0->SetLineWidth(3);
  h2_residual_x_sta3_layer0->SetLineWidth(3);
  h2_residual_x_sta3_layer0->SetLineColor(kBlack);
  h2_residual_x_sta3_layer0->Scale(h1_residual_x_sta3_layer0->Integral()/h2_residual_x_sta3_layer0->Integral());
  h2_residual_x_sta3_layer0->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta3_layer0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta3_layer0->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta3_layer0->SaveAs("../docs/"+outputname+"_com_res_sta3_layer0.png");

  TCanvas* c_res_com_sta3_layer1 = new TCanvas("c_res_com_sta3_layer1");
  h1_residual_x_sta3_layer1->Draw();
  h1_residual_x_sta3_layer1->GetXaxis()->SetTitle("residual_x_station3_layer1 (mm)");
  h1_residual_x_sta3_layer1->SetLineColor(94);
  h1_residual_x_sta3_layer1->SetFillColor(94);
  h1_residual_x_sta3_layer1->SetLineWidth(3);
  h2_residual_x_sta3_layer1->SetLineWidth(3);
  h2_residual_x_sta3_layer1->SetLineColor(kBlack);
  h2_residual_x_sta3_layer1->Scale(h1_residual_x_sta3_layer1->Integral()/h2_residual_x_sta3_layer1->Integral());
  h2_residual_x_sta3_layer1->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta3_layer1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta3_layer1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta3_layer1->SaveAs("../docs/"+outputname+"_com_res_sta3_layer1.png");

  TCanvas* c_res_com_sta3_layer2 = new TCanvas("c_res_com_sta3_layer2");
  h1_residual_x_sta3_layer2->Draw();
  h1_residual_x_sta3_layer2->GetXaxis()->SetTitle("residual_x_station3_layer2 (mm)");
  h1_residual_x_sta3_layer2->SetLineColor(94);
  h1_residual_x_sta3_layer2->SetFillColor(94);
  h1_residual_x_sta3_layer2->SetLineWidth(3);
  h2_residual_x_sta3_layer2->SetLineWidth(3);
  h2_residual_x_sta3_layer2->SetLineColor(kBlack);
  h2_residual_x_sta3_layer2->Scale(h1_residual_x_sta3_layer2->Integral()/h2_residual_x_sta3_layer2->Integral());
  h2_residual_x_sta3_layer2->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta3_layer2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta3_layer2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta3_layer2->SaveAs("../docs/"+outputname+"_com_res_sta3_layer2.png");


  TCanvas* c_res_com_sta1 = new TCanvas("c_res_com_sta1");
  h1_residual_x_sta1->Draw();
  h1_residual_x_sta1->GetXaxis()->SetTitle("residual_x_station1 (mm)");
  h1_residual_x_sta1->SetLineColor(94);
  h1_residual_x_sta1->SetFillColor(94);
  h1_residual_x_sta1->SetLineWidth(3);
  h2_residual_x_sta1->SetLineWidth(3);
  // h3_residual_x_sta1->SetLineWidth(3);
  h2_residual_x_sta1->SetLineColor(kBlack);
  // h3_residual_x_sta1->SetLineColor(kRed);
  h2_residual_x_sta1->Scale(h1_residual_x_sta1->Integral()/h2_residual_x_sta1->Integral());
  // h3_residual_x_sta1->Scale(h1_residual_x_sta1->Integral()/h3_residual_x_sta1->Integral());
  h2_residual_x_sta1->Draw("same");
  // h3_residual_x_sta1->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta1->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  // f1_gaus->SetLineColor(kRed);
  // h3_residual_x_sta1->Fit(f1_gaus,"L","",-0.05,0.05);
  // tex.SetTextColor(kRed);
  // tex.DrawLatex(0.15,0.55,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  // tex.DrawLatex(0.15,0.50,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta1->SaveAs("../docs/"+outputname+"_com_res_sta1.png");

  TCanvas* c_res_com_sta2 = new TCanvas("c_res_com_sta2");
  h1_residual_x_sta2->Draw();
  h1_residual_x_sta2->GetXaxis()->SetTitle("residual_x_station2 (mm)");
  h1_residual_x_sta2->SetLineColor(94);
  h1_residual_x_sta2->SetFillColor(94);
  h1_residual_x_sta2->SetLineWidth(3);
  h2_residual_x_sta2->SetLineWidth(3);
  // h3_residual_x_sta2->SetLineWidth(3);
  h2_residual_x_sta2->SetLineColor(kBlack);
  // h3_residual_x_sta2->SetLineColor(kRed);
  h2_residual_x_sta2->Scale(h1_residual_x_sta2->Integral()/h2_residual_x_sta2->Integral());
  // h3_residual_x_sta2->Scale(h1_residual_x_sta2->Integral()/h3_residual_x_sta2->Integral());
  h2_residual_x_sta2->Draw("same");
  // h3_residual_x_sta2->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta2->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  // f1_gaus->SetLineColor(kRed);
  // h3_residual_x_sta2->Fit(f1_gaus,"L","",-0.05,0.05);
  // tex.SetTextColor(kRed);
  // tex.DrawLatex(0.15,0.55,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  // tex.DrawLatex(0.15,0.50,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta2->SaveAs("../docs/"+outputname+"_com_res_sta2.png");

  TCanvas* c_res_com_sta3 = new TCanvas("c_res_com_sta3");
  h1_residual_x_sta3->Draw();
  h1_residual_x_sta3->GetXaxis()->SetTitle("residual_x_station3 (mm)");
  h1_residual_x_sta3->SetLineColor(94);
  h1_residual_x_sta3->SetFillColor(94);
  h1_residual_x_sta3->SetLineWidth(3);
  h2_residual_x_sta3->SetLineWidth(3);
  // h3_residual_x_sta3->SetLineWidth(3);
  h2_residual_x_sta3->SetLineColor(kBlack);
  // h3_residual_x_sta3->SetLineColor(kRed);
  h2_residual_x_sta3->Scale(h1_residual_x_sta3->Integral()/h2_residual_x_sta3->Integral());
  // h3_residual_x_sta3->Scale(h1_residual_x_sta3->Integral()/h3_residual_x_sta3->Integral());
  h2_residual_x_sta3->Draw("same");
  // h3_residual_x_sta3->Draw("same");
  leg->Draw();
  f1_gaus->SetLineColor(94);
  h1_residual_x_sta3->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(94);
  tex.DrawLatex(0.15,0.85,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.80,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  f1_gaus->SetLineColor(kBlack);
  h2_residual_x_sta3->Fit(f1_gaus,"L","",-0.05,0.05);
  tex.SetTextColor(kBlack);
  tex.DrawLatex(0.15,0.70,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  tex.DrawLatex(0.15,0.65,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  // f1_gaus->SetLineColor(kRed);
  // h3_residual_x_sta3->Fit(f1_gaus,"L","",-0.05,0.05);
  // tex.SetTextColor(kRed);
  // tex.DrawLatex(0.15,0.55,Form("mean = %.4f #pm %.4f ", f1_gaus->GetParameter(1), f1_gaus->GetParError(1)));
  // tex.DrawLatex(0.15,0.50,Form("sigma = %.4f #pm %.4f ", f1_gaus->GetParameter(2), f1_gaus->GetParError(2)));
  c_res_com_sta3->SaveAs("../docs/"+outputname+"_com_res_sta3.png");

  TCanvas* c_chi2_com = new TCanvas("c_chi2_com");
  h1_fit_chi2->Draw();
  h1_fit_chi2->GetXaxis()->SetTitle("Track chi2");
  h1_fit_chi2->SetLineColor(94);
  h1_fit_chi2->SetFillColor(94);
  h1_fit_chi2->SetLineWidth(3);
  h2_fit_chi2->SetLineWidth(3);
  // h3_fit_chi2->SetLineWidth(3);
  h2_fit_chi2->SetLineColor(kBlack);
  // h3_fit_chi2->SetLineColor(kRed);
  h2_fit_chi2->Scale(h1_fit_chi2->Integral()/h2_fit_chi2->Integral());
  // h3_fit_chi2->Scale(h1_fit_chi2->Integral()/h3_fit_chi2->Integral());
  h2_fit_chi2->Draw("same");
  // h3_fit_chi2->Draw("same");
  f1_landau->SetLineColor(94);
  h1_fit_chi2->Fit(f1_landau, "L","",0,30);
  // tex.SetTextColor(94);
  // tex.DrawLatex(0.65,0.55,Form("MPV = %.3f #pm %.3f ", f1_landau->GetParameter(1), f1_landau->GetParError(1)));
  f1_landau->SetLineColor(kBlack);
  h2_fit_chi2->Fit(f1_landau, "L","",0,30);
  // tex.SetTextColor(kBlack);
  // tex.DrawLatex(0.65,0.45,Form("MPV = %.3f #pm %.3f ", f1_landau->GetParameter(1), f1_landau->GetParError(1)));
  // f1_landau->SetLineColor(kRed);
  // h3_fit_chi2->Fit(f1_landau, "L","",0,30);
  // tex.SetTextColor(kRed);
  // tex.DrawLatex(0.65,0.35,Form("MPV = %.3f #pm %.3f ", f1_landau->GetParameter(1), f1_landau->GetParError(1)));
  leg->Draw();
  c_chi2_com->SaveAs("../docs/"+outputname+"_com_chi2.png");

  //momentum

  TCanvas* c_pz_com = new TCanvas("c_pz_com");
  h1_pz->Draw();
  h1_pz->GetXaxis()->SetTitle("Track Pz (GeV)");
  h1_pz->SetLineColor(94);
  h1_pz->SetFillColor(94);
  h1_pz->SetLineWidth(3);
  h2_pz->SetLineWidth(3);
  // h3_pz->SetLineWidth(3);
  h2_pz->SetLineColor(kBlack);
  // h3_pz->SetLineColor(kRed);
  h2_pz->Scale(h1_pz->Integral()/h2_pz->Integral());
  // h3_pz->Scale(h1_pz->Integral()/h3_pz->Integral());
  h2_pz->Draw("same");
  // h3_pz->Draw("same");
  leg->Draw();
  c_pz_com->SaveAs("../docs/"+outputname+"_com_pz.png");

  TCanvas* c_py_com = new TCanvas("c_py_com");
  h1_py->Draw();
  h1_py->GetXaxis()->SetTitle("Track Py (GeV)");
  h1_py->SetLineColor(94);
  h1_py->SetFillColor(94);
  h1_py->SetLineWidth(3);
  h2_py->SetLineWidth(3);
  h2_py->SetLineColor(kBlack);
  h2_py->Scale(h1_py->Integral()/h2_py->Integral());
  h2_py->Draw("same");
  leg->Draw();
  c_py_com->SaveAs("../docs/"+outputname+"_com_py.png");


  TCanvas* c_px_com = new TCanvas("c_px_com");
  h1_px->Draw();
  h1_px->GetXaxis()->SetTitle("Track Px (GeV)");
  h1_px->SetLineColor(94);
  h1_px->SetFillColor(94);
  h1_px->SetLineWidth(3);
  h2_px->SetLineWidth(3);
  h2_px->SetLineColor(kBlack);
  h2_px->Scale(h1_px->Integral()/h2_px->Integral());
  h2_px->Draw("same");
  leg->Draw();
  c_px_com->SaveAs("../docs/"+outputname+"_com_px.png");


  TCanvas* c_y_com = new TCanvas("c_y_com");
  h1_y->Draw();
  h1_y->GetXaxis()->SetTitle("Track y");
  h1_y->SetLineColor(94);
  h1_y->SetFillColor(94);
  h1_y->SetLineWidth(3);
  h2_y->SetLineWidth(3);
  h2_y->SetLineColor(kBlack);
  h2_y->Scale(h1_y->Integral()/h2_y->Integral());
  h2_y->Draw("same");
  leg->Draw();
  c_y_com->SaveAs("../docs/"+outputname+"_com_y.png");

  TCanvas* c_x_com = new TCanvas("c_x_com");
  h1_x->Draw();
  h1_x->GetXaxis()->SetTitle("Track x");
  h1_x->SetLineColor(94);
  h1_x->SetFillColor(94);
  h1_x->SetLineWidth(3);
  h2_x->SetLineWidth(3);
  h2_x->SetLineColor(kBlack);
  h2_x->Scale(h1_x->Integral()/h2_x->Integral());
  h2_x->Draw("same");
  leg->Draw();
  c_x_com->SaveAs("../docs/"+outputname+"_com_x.png");

  TCanvas* c_z_com = new TCanvas("c_momentum_com");
  h1_z->Draw();
  h1_z->GetXaxis()->SetTitle("Track momentum (GeV)");
  h1_z->SetLineColor(94);
  h1_z->SetFillColor(94);
  h1_z->SetLineWidth(3);
  h2_z->SetLineWidth(3);
  h2_z->SetLineColor(kBlack);
  h2_z->Scale(h1_z->Integral()/h2_z->Integral());
  h2_z->Draw("same");
  leg->Draw();
  c_z_com->SaveAs("../docs/"+outputname+"_com_momentum.png");

  TCanvas* c_tx_com = new TCanvas("c_tx_com");
  h1_tx->Draw();
  h1_tx->GetXaxis()->SetTitle("Track Tx");
  h1_tx->SetLineColor(94);
  h1_tx->SetFillColor(94);
  h1_tx->SetLineWidth(3);
  h2_tx->SetLineWidth(3);
  h2_tx->SetLineColor(kBlack);
  h2_tx->Scale(h1_tx->Integral()/h2_tx->Integral());
  h2_tx->Draw("same");
  leg->Draw();
  c_tx_com->SaveAs("../docs/"+outputname+"_com_tx.png");

  TCanvas* c_ty_com = new TCanvas("c_ty_com");
  h1_ty->Draw();
  h1_ty->GetXaxis()->SetTitle("Track Ty");
  h1_ty->SetLineColor(94);
  h1_ty->SetFillColor(94);
  h1_ty->SetLineWidth(3);
  h2_ty->SetLineWidth(3);
  h2_ty->SetLineColor(kBlack);
  h2_ty->Scale(h1_ty->Integral()/h2_ty->Integral());
  h2_ty->Draw("same");
  leg->Draw();
  c_ty_com->SaveAs("../docs/"+outputname+"_com_ty.png");

  }
