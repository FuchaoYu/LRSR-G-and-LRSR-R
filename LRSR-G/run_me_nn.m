clear

% src_str = {'Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
% tgt_str = {'amazon','webcam','dslr','Caltech10','webcam','dslr','Caltech10','amazon','dslr','Caltech10','amazon','webcam'};
src_str = {'SF0'};
tgt_str = {'SF1'};


for i = 1:length(tgt_str)
    src = src_str{i};
    tgt = tgt_str{i};
    fprintf(' %s vs %s ', src, tgt);
%实验
    p=[0.0001,0.001,0.01,0.1,1,10];
    q=[0.0001,0.001,0.01,0.1,1,10];
    j=[0.0001,0.001,0.01,0.1,1,10];
    lambda=0.1;


    for o = 1: length(p)
        for v = 1: length(q)
            for m = 1: length(j)
                for s = 1: length(lambda)
                        load(['CWRU\' src '.mat']); 
                        Xs = fits';
                        Xs_label = lab;
                        clear fits;
                        clear lab;

                        load(['CWRU\' tgt '.mat']); 
                        Xt = fits';
                        Xt_label = lab;
                        clear fits;
                        clear lab;
                        % ------------------------------------------
                        %             Transfer Learning
                        % ------------------------------------------
                        Xs = Xs./repmat(sqrt(sum(Xs.^2)),[size(Xs,1) 1]); 
                        Xt = Xt./repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);        
                        P = TSL_LRSR(Xs,Xt,Xs_label,p(o),q(v),j(m),lambda(i));
                        X_train = P'*Xs;
                        Y_test  = P'*Xt; 
                        % -------------------------------------------
                        %               Classification
                        % -------------------------------------------
                        X_train = X_train./repmat(sqrt(sum(X_train.^2)),[size(X_train,1) 1]); 
                        Y_test  = Y_test ./repmat(sqrt(sum(Y_test.^2)),[size(Y_test,1) 1]); 
                    %   cls = knnclassify(Y_test',X_train',Xs_label);
                        knn_model = fitcknn(X_train',Xs_label,'NumNeighbors',1);
                        cls = knn_model.predict(Y_test');
                        accl(s) = sum(cls==Xt_label)/length(Xt_label)*100;
                end
                acclm(m,:)=accl;
                [accl_max(m), r] = max(accl); 
            end
            accjm(:,:,v)=acclm;
            accj = accl_max;
            [accj_max(v), n] = max(accj);       
        end
        accqm(:,:,:,o)=accjm;
        accq = accj_max;
        [accq_max(o),f] = max(accq);
    end
    %[accj_max, n] = max(accj);
    accp = accq_max;
    [accp_max,e] = max(accp);
    fprintf(' %2.2f%%,%1.3f,%1.3f,%1.3f,%1.3f\n',accp_max,p(e),q(f),j(n),lambda(r));
end