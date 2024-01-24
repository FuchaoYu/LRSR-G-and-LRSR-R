function P = TSL_LRSR(Xs,Xt,Xs_label,alpha,beta,gamma,lambda)

Y = Construct_Y(Xs_label,length(Xs_label)); 
B = Construct_B(Y);
Class = length(unique(Xs_label));
Max_iter = 100;
[m,n1] = size(Xs); n2 = size(Xt,2);
max_mu = 10^6;
mu = 0.1;
rho = 1.01;
convergence = 10^-6;
options = [];
options.ReducedDim = Class;
[P1,~] = PCA1(Xs', options);
% ----------------------------------------------
%               Initialization
% ----------------------------------------------
M = ones(Class,n1);
P = zeros(m,Class);
E = zeros(Class,n2);
N = zeros(Class,n2);

Z = zeros(n1,n2);
Z1 = zeros(n1,n2);
Z2 = zeros(n1,n2);

Y1 = zeros(Class,n2);
Y2 = zeros(n1,n2);
Y3 = zeros(n1,n2);
% ------------------------------------------------
%                   Main Loop
% ------------------------------------------------
for iter = 1:Max_iter 
    
    % updating P
    V0 = Y+B.*M;
    V1 = Xt-Xs*Z;
    V2 = E+N-Y1/mu;
    if (iter == 1)
        P = P1;
    else
        P = (2*Xs*Xs'+mu*V1*V1'+2*lambda*eye(m))\(2*Xs*V0'+mu*V1*V2');
%         Q = Xs*V0'+mu*V1*V2';
%         [U,~,V] = mySVD(Q);
%         P = U*V'; 
    end       
    
    % updating M
    R = P'*Xs-Y;
    gp = B.*R;
    [numm1,numm2] = size(gp);
    for jk1 = 1:numm1
        for jk2 = 1:numm2
            M(jk1,jk2) = max(gp(jk1,jk2),0);
        end
    end    
    % updating E    
    the2 = beta/mu;
    temp_E = P'*Xt-P'*Xs*Z-N+Y1/mu;
    E = max(0,temp_E-the2)+min(0,temp_E+the2);    
    
    % updating Z
    V3 = Z1-Y2/mu;
    V4 = Z2-Y3/mu;
    V5 = P'*Xt-E-N+Y1/mu;
    Z = (2*eye(n1)+Xs'*P*P'*Xs)\(V3+V4+Xs'*P*V5);   
    
    % updating  Z1
    ta = 1/mu;
    temp_Z1 = Z+Y2/mu;
    [U01,S01,V01] = svd(temp_Z1,'econ');
    S01 = diag(S01);
    svp = length(find(S01>ta));
    if svp >= 1
        S01 = S01(1:svp)-ta;
    else
        svp = 1;
        S01 = 0;    
    end
    Z1 = U01(:,1:svp)*diag(S01)*V01(:,1:svp)';
    
    % updating Z2
    taa = alpha/mu;
    temp_Z2 = Z+Y3/mu;
    Z2 = max(0,temp_Z2-taa)+min(0,temp_Z2+taa);
    
    % updating N
    N = mu/(2*gamma+mu)*(-P'*Xs*Z-E+P'*Xt+Y1/mu);
    
    % updating Y1, Y2, Y3
    Y1 = Y1+mu*(P'*Xt-P'*Xs*Z-E-N);
    Y2 = Y2+mu*(Z-Z1);
    Y3 = Y3+mu*(Z-Z2);
    
    % updating mu
    mu = min(rho*mu,max_mu);
    
    % checking convergence
    leq1 = norm(P'*Xt-P'*Xs*Z-E-N,Inf);
    leq2 = norm(Z-Z1,Inf);
    leq3 = norm(Z-Z2,Inf);
    if iter > 2
         if leq1<convergence && leq2<convergence && leq3<convergence
              break
         end
     end      
end

end

function B = Construct_B(Y)
%%
B = Y;
B(Y==0) = -1;
end

function Y = Construct_Y(gnd,num_l)
%%
% gnd:标签向量；
% num_l:表示有标签样本的数目；
% Y:生成的标签矩阵；
nClass = length(unique(gnd));
Y = zeros(nClass,length(gnd));
for i = 1:num_l
    for j = 1:nClass
        if j == gnd(i)
            Y(j,i) = 1;
        end  
    end
end
end