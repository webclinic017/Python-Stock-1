#  imports

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.compat import lzip
from scipy.stats import f
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#from mpl_toolkits.mplot3d import Axes3D

class LinearRegressionResidualPlot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self):
        linear_model = sm.OLS(self.y, sm.add_constant(self.x)).fit()

        return linear_model

    @staticmethod
    def check_linearity_assumption(fitted_y, residuals):
        qq = ProbPlot(residuals)
        plot_1 = plt.figure()
        plot_1.axes[0] = sns.residplot(fitted_y, residuals,
                                       lowess=True,
                                       scatter_kws={'alpha': 0.5},
                                       line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

        plot_1.axes[0].set_title('Residuals vs Fitted')
        #plot_1.axes[0].set_xlim(min(residuals) - 0.1, max(residuals) + 0.01)
        plot_1.axes[0].set_xlabel('Fitted values')
        plot_1.axes[0].set_ylabel('Residuals')

        # annotations
        '''
        norm_resid = np.flip(np.argsort(residuals), 0)

        s_thresh = max(qq.theoretical_quantiles)
        norm_resid_top = pd.DataFrame(residuals).index[abs(residuals)>s_thresh].to_list()

        for i in norm_resid_top:
            plot_1.axes[0].annotate(fitted_y.index[i],
                                    xy=(fitted_y[i],
                                        residuals[i]))
        '''

        plt.savefig("ResVsFitted.png")

    @staticmethod
    def check_residual_normality(fitted_y, residuals_normalized):
        qq = ProbPlot(residuals_normalized)
        plot_2 = qq.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
        plot_2.axes[0].set_title('Normal Q-Q')
        plot_2.axes[0].set_xlabel('Theoretical Quantiles')
        plot_2.axes[0].set_ylabel('T Student Standardized Residuals')

        # annotations

        df = pd.DataFrame(pd.DataFrame(residuals_normalized).set_index(fitted_y.index))
        df['ranks'] = df.rank(method='dense').astype(int)-1
        ranks = df.ranks.values

        s_thresh = max(qq.theoretical_quantiles)
        abs_norm_resid_top = pd.DataFrame(residuals_normalized).index[abs(residuals_normalized)>s_thresh].to_list()

        for r, i in enumerate(abs_norm_resid_top):
            plot_2.axes[0].annotate(fitted_y.index[i],
                                    xy=(qq.theoretical_quantiles[ranks[i]],residuals_normalized[i]))

        plt.savefig("Normality.png")

    @staticmethod
    def check_homoscedacticity(fitted_y, residuals_normalized):

        qq = ProbPlot(residuals_normalized)
        s_thresh = np.sqrt(max(qq.theoretical_quantiles))

        # absolute squared normalized residuals
        residuals_norm_abs_sqrt = np.sqrt(np.abs(residuals_normalized))

        plot_3 = plt.figure()
        plt.scatter(fitted_y, residuals_norm_abs_sqrt, alpha=0.5)
        sns.regplot(fitted_y, residuals_norm_abs_sqrt,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plot_3.axes[0].set_title('Scale-Location')
        plot_3.axes[0].set_xlabel('Fitted values')
        plot_3.axes[0].set_ylabel("$\\sqrt{|T Student Standardized Residuals|}$")
        plot_3.axes[0].axhline(s_thresh, ls='--', color='black')
        
        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residuals_norm_abs_sqrt), 0)

        abs_sq_norm_resid_top = pd.DataFrame(residuals_norm_abs_sqrt).index[residuals_norm_abs_sqrt>s_thresh].to_list()
        for i in abs_sq_norm_resid_top:
            plot_3.axes[0].annotate(fitted_y.index[i],
                                    xy=(fitted_y[i],
                                        residuals_norm_abs_sqrt[i]))
        plt.savefig("Homoscadasticity.png")

    @staticmethod
    def check_influence(fitted_y, linear_model, cooks, leverage, residuals_normalized):
        qq = ProbPlot(residuals_normalized)

        cooks = np.round(f.pdf(cooks,len(linear_model.tvalues)+1, len(linear_model.fittedvalues)-len(linear_model.tvalues)-1),2)

        c_thresh = .1
        l_thresh = (2*(len(linear_model.tvalues)-1)/len(linear_model.fittedvalues))
        s_thresh = max(qq.theoretical_quantiles)
        
        c_thresh1 = .1
        c_thresh2 = .5

        l_thresh1 = (2*(len(linear_model.tvalues)-1)/len(linear_model.fittedvalues))
        l_thresh2 = (3*(len(linear_model.tvalues)-1)/len(linear_model.fittedvalues))        
        
        plot_4 = plt.figure()
        plt.scatter(leverage, residuals_normalized, alpha=0.5)
        sns.regplot(leverage, residuals_normalized,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        #plot_4.axes[0].set_xlim(0, max(leverage) + 0.01)
        plot_4.axes[0].axvline(l_thresh1, ls='--', color='blue')
        plot_4.axes[0].axvline(l_thresh2, ls='--', color='red')
        plot_4.axes[0].axhline(s_thresh, ls='--', color='black')
        plot_4.axes[0].axhline(-s_thresh, ls='--', color='black')
        #plot_4.axes[0].set_ylim(min(residuals_normalized)*1.05, max(residuals_normalized)*1.05)
        plot_4.axes[0].set_title('T Student Residuals vs Leverage w/ F Dist (C:)ooks %')
        plot_4.axes[0].set_xlabel('Leverage')
        plot_4.axes[0].set_ylabel('T Student Residuals')

        labels_ = fitted_y.index

        x = cooks
        y = leverage
        z = residuals_normalized

        outlier_check = pd.concat([pd.DataFrame(x),pd.DataFrame(y),pd.DataFrame(z)],axis=1).set_index(labels_)

        outlier_check.columns =  ['cooks', 'leverage', 'tsres']

        flag = []

        for i in range(0,len(outlier_check)):
            if( (outlier_check.iloc[i][0] >= c_thresh) or (outlier_check.iloc[i][1] >= l_thresh) or (abs(outlier_check.iloc[i][2]) >= s_thresh) ):
                #print(outlier_check.iloc[i])
                #print()
                flag.append(True)
            else:
                flag.append(False)

        outlier_check = pd.concat([outlier_check,pd.DataFrame(flag).set_index(labels_)],axis=1)

        outlier_check.columns =  ['cooks', 'leverage', 'tsres', 'flagged']        
        
        # annotations
        search = outlier_check[outlier_check['flagged']==1].index.to_list()

        influence_top = []

        for i in search:
            v = outlier_check.index.to_list().index(i)
            influence_top.append(v)
                
        for i in influence_top:       

            CString = ""
            cc = 'black'
            
            if(outlier_check.iloc[i][0]>=c_thresh2):
                cc='red'
                CString = CString + " (C:" + str(outlier_check.iloc[i][0].round(2)) + ")"
            elif(outlier_check.iloc[i][0]>=c_thresh1):
                cc='blue'
                CString = CString + " (C:" + str(outlier_check.iloc[i][0])
            
            plot_4.axes[0].annotate(fitted_y.index[i] + CString, color=cc,
                                xy=(leverage[i],
                                    residuals_normalized[i]))
            
        plt.savefig("Influence.png")

    def diagnostic_plots(self, linear_model):
        """

        :param linear_model: Linear Model Fit on the Data
        :return: None

        This method validates the assumptions of Linear Model
        """
        diagnostic_result = {}

        summary = linear_model.summary()
        #diagnostic_result['summary'] = str(summary)

        # fitted values
        fitted_y = linear_model.fittedvalues
        # model residuals
        residuals = linear_model.resid

        # normalized residuals
        residuals_normalized = linear_model.get_influence().resid_studentized_internal

        # leverage, from statsmodels internals
        leverage = linear_model.get_influence().hat_matrix_diag

        # cook's distance, from statsmodels internals
        cooks = linear_model.get_influence().cooks_distance[0]

        self.check_linearity_assumption(fitted_y, residuals)

        self.check_residual_normality(fitted_y, residuals_normalized)

        self.check_homoscedacticity(fitted_y, residuals_normalized)

        self.check_influence(fitted_y, linear_model, cooks, leverage, residuals_normalized)

        # 1. Non-Linearity Test
        try:
            name = ['F value', 'p value']
            test = sms.linear_harvey_collier(linear_model)
            linear_test_result = lzip(name, test)
        except Exception as e:
            linear_test_result = str(e)
        diagnostic_result['Non_Linearity_Test'] = linear_test_result

        # 2. Hetroskedasticity Test
        name = ['Lagrange multiplier statistic', 'p-value',
                'f-value', 'f p-value']
        test = sms.het_breuschpagan(linear_model.resid, linear_model.model.exog)
        test_val = lzip(name, test)
        diagnostic_result['Hetroskedasticity_Test'] = test_val

        # 3. Normality of Residuals
        name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
        test = sms.jarque_bera(linear_model.resid)
        test_val = lzip(name, test)
        diagnostic_result['Residual_Normality_Test'] = test_val

        # 4. MultiCollnearity Test
        test = np.linalg.cond(linear_model.model.exog)
        test_val = [('condition no',test)]
        diagnostic_result['MultiCollnearity_Test'] = test_val

        # 5. Residuals Auto-Correlation Tests
        test = sms.durbin_watson(linear_model.resid)
        test_val = [('p value', test)]
        diagnostic_result['Residual_AutoCorrelation_Test'] = test_val

        json_result = json.dumps(diagnostic_result)
        return summary, json_result

