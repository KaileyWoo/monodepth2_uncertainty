2020-10-26 (*all eigen_zhou data, batch size 12*)
     abs_rel    |      sq_rel      |     rmse     |    rmse_log       |       a1             |       a2         |       a3       | 
&     0.116    &       0.917     &   4.854      &   0.193               &   0.876        &   0.958     &   0.980  \\     eigen
&     0.092    &       0.560     &   3.962      &   0.138               &   0.913        &   0.982     &   0.994  \\      eigen_benchmark
&     0.089    &       0.525     &   3.867      &   0.135               &   0.916        &   0.983     &   0.995  \\     eigen_benchmark    --post_process


2020-11-24 (part of eigen_zhou data, batch size 12)
Scaling ratios | med: 37.316 | std: 0.088
abs_rel         |   sq_rel       |     rmse      |  rmse_log    |       a1         |       a2          |       a3         | 
&   0.119      &   0.887       &   4.783     &   0.193          &   0.868     &   0.958     &   0.981      \\


2020-12-12(part of eigen_zhou data, batch size 7)
Scaling ratios | med: 37.813 | std: 0.080
   abs_re   l   |   sq_rel     |     rmse     | rmse_log      |       a1           |       a2            |       a3          | 
&   0.118      &   0.878     &   4.817     &   0.193          &   0.868       &   0.958        &   0.982      \\


2020-12-11(use teacher-student)(part of eigen_zhou data, batch size 7, source scale)
Scaling ratios | med: 56.877 | std: 0.100
   abs_rel      |   sq_rel      |     rmse      | rmse_log      |       a1         |        a2           |       a3          | 
&   0.130      &   0.906      &   4.852     &   0.197           &   0.845     &    0.954      &   0.982       \\


2020-12-16(use teacher-student)(part of eigen_zhou data, batch size 7, scale)   weights_17
Scaling ratios | med: 57.767 | std: 0.099 
   abs_rel      |   sq_rel    |     rmse     | rmse_log      |         a1           |       a2           |       a3           | 
&   0.131      &   0.929    &   4.794    &   0.197           &      0.845      &   0.955      &   0.982        \\


2020-12-18(use teacher-student)(part of eigen_zhou data, batch size 7, source_scale, uncertLoss0.05)  weights_18
 Scaling ratios | med: 58.223 | std: 0.101
   abs_rel      |   sq_rel       |     rmse        | rmse_log       |       a1           |       a2             |       a3         | 
&   0.127      &   0.874       &   4.759       &   0.195            &   0.850       &   0.956        &   0.983      \\


2020-12-21(use teacher-student)(all eigen_zhou data, batch size 7, source_scale, uncertLoss0.05)
(There are 39810 training items and 4424 validation items)
Scaling ratios | med: 57.327 | std: 0.105
   abs_rel       |   sq_rel       |     rmse        | rmse_log       |       a1           |       a2          |       a3           | 
&   0.122       &   0.837       &   4.772        &   0.193           &   0.856       &   0.957     &   0.983        \\

2021-03-05(use teacher-student)(part of eigen_zhou data, batch size 7, source_scale, No uncerLoss)
Scaling ratios | med: 39.159 | std: 0.087
   abs_rel      |   sq_rel      |     rmse        | rmse_log      |       a1          |       a2          |       a3        | 
&   0.120      &   0.856      &   4.829       &   0.194           &   0.862      &   0.958     &   0.982     \\



------------------------------------------------------------------------------------------------------------------------------------------------------------
2021-5-14
part of eigen_zhou data, batch size 7, learning_rate 1e-5
" pred_uncerts = torch.exp(outputs[("uncert", 0)]) "
" reprojection_losses.append(reprojection_loss/pred_uncerts +  torch.exp(pred_uncerts)) "
Scaling ratios | med: 65.587 | std: 0.117
        abs_rel       |       sq_rel     |     rmse        |     rmse_log      |        a1           |         a2          |          a3           | 
&        0.152       &      1.059      &   5.152       &        0.214          &    0.804       &     0.942     &       0.980      \\



2021-5-17
part of eigen_zhou data,  batch size  = 12,  learning_rate = 1e-4,  uncert = True,  dropout = False
" pred_uncerts = torch.exp(outputs[("uncert", 0)]) "
" reprojection_losses.append(reprojection_loss/pred_uncerts +  torch.exp(pred_uncerts)) "
Scaling ratios | med: 35.083 | std: 0.087
        abs_rel      |      sq_rel      |       rmse      |       rmse_log     |       a1        |       a2        |       a3        | 
&      0.118        &      0.883      &     4.764     &           0.192       &   0.869    &   0.958    &   0.981     \\           --post_process

                abs_rel          |                  rmse          |                       a1           | 
      AUSE |     AURG     |     AUSE |     AURG   |     AUSE |     AURG    | 
&   0.068  &   0.010   &   3.802  &   0.236    &   0.102  &   0.017    \\                         --post_process




2021-5-19
part of eigen_zhou data,  batch size  = 12,  learning_rate = 1e-4,  uncert = True,  dropout = True
Scaling ratios | med: 63.167 | std: 0.098
     abs_rel     |       sq_rel      |        rmse      |     rmse_log     |          a1         |         a2          |           a3          | 
&     0.136     &       1.041      &      4.872     &       0.201          &     0.841      &     0.951      &       0.981      \\



2021-5-20
part of eigen_zhou data,  batch size  = 7,  learning_rate = 1e-5,  uncert = True,  dropout = True
使用5-19的网络获取不确定性来计算loss函数，for 循环3次
"reprojection_losses.append(reprojection_loss/(uncerts + 1e-10) +  torch.exp(uncerts))"
Scaling ratios | med: 63.749 | std: 0.102
        abs_rel     |   sq_rel       |          rmse       |      rmse_log       |        a1          |        a2        |        a3             | 
&      0.144       &   0.990       &        5.074      &        0.208           &    0.818      &    0.945    &   0.981          \\




2021-5-29
part of eigen_zhou data,  batch size  = 12,  learning_rate = 1e-4,  uncert = True,  dropout = True
使用5-19的网络获取不确定性来计算loss函数，for 循环8次
"reprojection_losses.append(reprojection_loss/(uncerts + 1e-10) +  torch.exp(uncerts))"
Scaling ratios | med: 62.978 | std: 0.089
       abs_rel      |      sq_rel      |       rmse      |       rmse_log      |       a1          |          a2       |           a3         | 
&      0.131       &      0.898      &     4.865      &         0.199          &   0.842     &     0.950    &       0.982     \\

------------------------------------------------------------------------------------------------------------------------------------------------------------

2021-6-18
part of eigen_zhou data,  batch size  = 12,  learning_rate = 1e-4,  uncert = True,  dropout = True
uncerts 是通过计算循环自己网络8次所得的output的方差(训练时)
"reprojection_losses.append(reprojection_loss/(uncerts + 1e-10) +  torch.exp(uncerts))"
估计时候：for循环了64次     
 Scaling ratios | med: 50.291 | std: 0.098
   abs_rel     |      sq_rel     |     rmse     |      rmse_log      |       a1         |       a2          |       a3         | 
&   0.121     &      0.848     &   4.759     &        0.190         &   0.861      &   0.958     &   0.983      \\     ( --post_process)

                  abs_rel             |                rmse                  |                   a1                     | 
      AUSE      |     AURG     |     AUSE     |     AURG     |     AUSE      |     AURG     | 
&   0.054     &   0.026     &   2.220     &   1.863      &   0.077       &   0.051      \\
&     low       &      high    &      low      &      high     &      low       &     high


2021-7-19
part of eigen_zhou data,  3*batchsize=4,  learning_rate = 1e-4,  uncert = False,  dropout = True
uncerts 是通过计算循环自己网络 11 次所得的output的方差(训练时)
"reprojection_losses.append(reprojection_loss/(uncerts + 1e-10) +  torch.exp(uncerts))"
估计时候：for循环了64次   
 Scaling ratios | med: 47.079 | std: 0.092
      abs_rel        |      sq_rel      |     rmse      |    rmse_log      |        a1          |       a2         |         a3        | 
&     0.117         &      0.821      &   4.717     &      0.189           &    0.867     &   0.960     &     0.983     \\         --post_process

                    abs_rel     |              rmse            |                 a1                | 
      AUSE |     AURG    |     AUSE |     AURG |     AUSE |     AURG   | 
&   0.047  &   0.032  &   2.082  &   2.000  &   0.062  &   0.062   \\





2021-6-22
使用梯度累加，3次。其它的都按照原来monodepth2来，结果显示和原来的差别不大，就是损失跳变比原来的大
part of eigen_zhou data,  batch size  = 4,  learning_rate = 1e-4
Scaling ratios | med: 34.742 | std: 0.084
   abs_rel      |   sq_rel         |     rmse        |    rmse_log           |          a1          |          a2          |       a3             | 
&   0.119      &   0.947         &   4.899       &        0.194              &      0.870     &      0.957       &   0.981         \\




2021-7-08
使用梯度累加，3*batchsize=4，learning_rate = 1e-4,  uncert = True,  dropout = False
使用已经训练好的7个boot网络计算不确定性
"reprojection_losses.append(reprojection_loss/(uncerts + 1e-10) +  torch.exp(uncerts))"
评价指标估计时，使用的pred_uncert = torch.exp(output[("uncert", 0)])
 Scaling ratios | med: 34.714 | std: 0.085
   abs_rel      |   sq_rel      |     rmse       | rmse_log    |       a1            |       a2         |           a3 | 
&   0.116      &   0.893      &   4.791      &   0.190         &   0.875       &   0.959     &       0.982  \\             --post_process

               abs_rel             |                   rmse              |                   a1                  | 
      AUSE   |     AURG     |     AUSE    |     AURG     |      AUSE      |     AURG | 
&   0.066  &   0.010    &    3.752    &   0.328     &     0.094      &    0.020  \\                          --post_process




2021-7-12
使用已经训练好的9个monodepth子网络计算精度和不确定性  
 Scaling ratios | med: 36.994 | std: 0.083
         abs_rel     |      sq_rel       |       rmse      |      rmse_log      |       a1         |       a2        |          a3        | 
&       0.113       &       0.795      &     4.610     &         0.186         &   0.876     &   0.961     &      0.983    \\     --post_process

             abs_rel            |                rmse            |                   a1            | 
      AUSE |     AURG    |     AUSE |     AURG   |     AUSE |     AURG | 
&   0.064  &   0.010  &   3.650  &   0.260   &   0.095  &   0.018  \\      --post_process




2021-7-10
使用梯度累加，3*batchsize=4，learning_rate = 1e-4,  uncert = False,  dropout = False
同时训练8个boot网络计算不确定性
"reprojection_losses.append(reprojection_loss/(uncerts + 1e-10) +  torch.exp(uncerts))"
 Scaling ratios | med: 57.139 | std: 0.097
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.123  &   0.798  &   4.925  &   0.194  &   0.851  &   0.954  &   0.983  \\

   abs_rel |          |     rmse |          |       a1 |          | 
      AUSE |     AURG |     AUSE |     AURG |     AUSE |     AURG | 
&   0.081  &  -0.002  &   3.195  &   0.969  &   0.139  &  -0.005  \\




2021-7-15 ( *all eigen_zhou data* )
使用梯度累加，3*batchsize=4，learning_rate = 1e-4,  uncert = True,  dropout = False
使用已经训练好的8个boot网络计算不确定性
Scaling ratios | med: 30.233 | std: 0.086
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.112  &   0.865  &   4.757  &   0.189  &   0.881  &   0.960  &   0.981  \\          --post_process

   abs_rel |          |     rmse |          |       a1 |          | 
      AUSE |     AURG |     AUSE |     AURG |     AUSE |     AURG | 
&   0.062  &   0.011  &   3.662  &   0.396  &   0.089  &   0.021  \\        --post_process


------------------------------------------------------------------------------------------------------------------------------------------------------------

2021-8-6(part eigen_zhou data )(self_teacher),     bachsize=12, learning_rate=1e-4, uncert=True, dropout=False
teacher网络为 “model_fullData_1026”，student网络 “uncert=torch.exp(outputs[("uncert", scale)])”"uncert_loss = L1_loss/(uncert + 1e-10) + uncert"
Scaling ratios | med: 30.319 | std: 0.086
    abs_rel |   sq_rel  |     rmse   | rmse_log |       a1   |       a2      |       a3      | 
&   0.112  &   0.833  &   4.729  &   0.188  &   0.877  &   0.960  &   0.983  \\  --eigen   
&   0.112  &   0.818  &   4.698  &   0.187  &   0.878  &   0.961  &   0.983  \\  --eigen                            --post_process

&   0.090  &   0.513  &   3.871  &   0.135  &   0.914  &   0.983  &   0.996  \\--eigen_benchmark
&   0.089  &   0.504  &   3.843  &   0.134  &   0.915  &   0.983  &   0.996  \\--eigen_benchmark --post_process

             abs_rel            |              rmse             |                a1                 | 
     AUSE    |   AURG    |    AUSE   |   AURG   |   AUSE    |    AURG   | 
&   0.034  &   0.040  &   2.254  &   1.775  &   0.035  &   0.078  \\  --eigen   
&   0.062  &   0.011  &   3.632  &   0.367  &   0.091  &   0.021  \\  --eigen                            --post_process

&   0.030  &   0.027  &   2.043  &   1.253  &   0.030  &   0.050  \\--eigen_benchmark
&   0.049  &   0.008  &   3.021  &   0.252  &   0.066  &   0.013  \\--eigen_benchmark --post_process

------------------------------------------------------------------------------------------------------------------------------------------------------------


2021-8-7(part eigen_zhou data )(self_teacher),     bachsize=12, learning_rate=1e-4, uncert=True, dropout=False
同上（2021-8-6），只是L1损失改为L2损失
Scaling ratios | med: 30.646 | std: 0.088
         abs_rel       |      sq_rel       |        rmse       |      rmse_log      |          a1        |          a2          |         a3          | 
&        0.113        &      0.806       &       4.681     &          0.187         &     0.876     &      0.961     &      0.983      \\     --post_process

                      abs_rel           |                         rmse               |                          a1                   | 
      AUSE      |       AURG     |       AUSE       |       AURG     |       AUSE       |      AURG      | 
&   0.035     &      0.039     &      2.167     &      1.852     &      0.036      &     0.079      \\

------------------------------------------------------------------------------------------------------------------------------------------------------------

2021-8-9( *all eigen_zhou data* )(self_teacher),     bachsize=12, learning_rate=1e-4, uncert=True, dropout=False
teacher网络为 “model_fullData_1026”，student网络 “uncert=torch.exp(outputs[("uncert", scale)])”"uncert_loss = L1_loss/(uncert + 1e-10) + uncert"
Scaling ratios | med: 29.269 | std: 0.090
    abs_rel |   sq_rel  |     rmse   | rmse_log |       a1   |       a2      |       a3      | 
&   0.111  &   0.845  &   4.723  &   0.188  &   0.881  &   0.960  &   0.982  \\  --eigen   
&   0.111  &   0.825  &   4.678  &   0.187  &   0.882  &   0.961  &   0.982  \\  --eigen                            --post_process

&   0.088  &   0.512  &   3.839  &   0.133  &   0.919  &   0.983  &   0.995  \\  --eigen_benchmark
&   0.087  &   0.497  &   3.790  &   0.132  &   0.919  &   0.984  &   0.996  \\  --eigen_benchmark --post_process

             abs_rel            |              rmse             |                a1                 | 
     AUSE    |   AURG    |    AUSE   |   AURG   |   AUSE    |    AURG   |  
&   0.034  &   0.039  &   2.200  &   1.833  &   0.034  &   0.075  \\  --eigen
&   0.062  &   0.011  &   3.639  &   0.351  &   0.088  &   0.020  \\  --eigen                            --post_process

&   0.029  &   0.027  &   1.947  &   1.333  &   0.028  &   0.047  \\  --eigen_benchmark
&   0.047  &   0.008  &   2.940  &   0.293  &   0.062  &   0.014  \\  --eigen_benchmark --post_process


------------------------------------------------------------------------------------------------------------------------------------------------------------

2021-8-18(part eigen_zhou data )(self_teacher),     bachsize=12, learning_rate=1e-4, uncert=True, dropout=False
teacher网络为 “model_fullData_1026”，student网络 “uncert=torch.exp(outputs[("uncert", 0)])”"uncert_loss = L1_loss/(uncert + 1e-10) + uncert"
与（2021-8-6）不同之处：计算 "单尺度" 上的损失 scale=0
Scaling ratios | med: 30.579 | std: 0.086
    abs_rel |   sq_rel  |     rmse   | rmse_log |       a1   |       a2      |       a3      | 
&   0.112  &   0.844  &   4.753  &   0.189  &   0.879  &   0.960  &   0.982  \\  --eigen
&   0.111  &   0.811  &   4.691  &   0.187  &   0.880  &   0.961  &   0.982  \\  --eigen                            --post_process

&   0.090  &   0.524  &   3.938  &   0.136  &   0.914  &   0.982  &   0.995  \\  --eigen_benchmark
&   0.089  &   0.504  &   3.878  &   0.135  &   0.915  &   0.983  &   0.995  \\  --eigen_benchmark --post_process

             abs_rel            |              rmse             |                a1                 | 
     AUSE    |   AURG    |    AUSE   |   AURG   |   AUSE    |    AURG   | 
&   0.033  &   0.041  &   1.967  &   2.088  &   0.032  &   0.080  \\  --eigen
&   0.062  &   0.011  &   3.597  &   0.402  &   0.088  &   0.022  \\  --eigen                            --post_process

&   0.029  &   0.029  &   1.791  &   1.576  &   0.028  &   0.052  \\  --eigen_benchmark
&   0.049  &   0.008  &   3.016  &   0.294  &   0.066  &   0.013  \\  --eigen_benchmark --post_process

------------------------------------------------------------------------------------------------------------------------------------------------------------



