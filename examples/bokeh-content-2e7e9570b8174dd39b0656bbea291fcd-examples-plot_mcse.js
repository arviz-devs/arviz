(function() {
  const fn = function() {
    (function(root) {
      function now() {
        return new Date();
      }
    
      const force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
    
    const element = document.getElementById("22991ffe-162d-4456-a064-42a5000c9192");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '22991ffe-162d-4456-a064-42a5000c9192' but no matching script tag was found.")
        }
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error(url) {
          console.error("failed to load " + url);
        }
    
        for (let i = 0; i < css_urls.length; i++) {
          const url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        for (let i = 0; i < js_urls.length; i++) {
          const url = js_urls[i];
          const element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.async = false;
          element.src = url;
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      const js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-3.0.2.min.js"];
      const css_urls = [];
    
      const inline_js = [    function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        function(Bokeh) {
          (function() {
            const fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                  const docs_json = '{"5fcc0ffa-34d2-4d60-ac98-41d8a3fadee0":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p61823","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p61822","attributes":{"tools":[{"type":"object","name":"ToolProxy","id":"p61814","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p61655"},{"type":"object","name":"ResetTool","id":"p61707"}]}},{"type":"object","name":"ToolProxy","id":"p61815","attributes":{"tools":[{"type":"object","name":"PanTool","id":"p61656"},{"type":"object","name":"PanTool","id":"p61708"}]}},{"type":"object","name":"ToolProxy","id":"p61816","attributes":{"tools":[{"type":"object","name":"BoxZoomTool","id":"p61657","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p61658","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"BoxZoomTool","id":"p61709","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p61710","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}}]}},{"type":"object","name":"ToolProxy","id":"p61817","attributes":{"tools":[{"type":"object","name":"WheelZoomTool","id":"p61659"},{"type":"object","name":"WheelZoomTool","id":"p61711"}]}},{"type":"object","name":"ToolProxy","id":"p61818","attributes":{"tools":[{"type":"object","name":"LassoSelectTool","id":"p61660","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p61661","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"LassoSelectTool","id":"p61712","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p61713","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}}]}},{"type":"object","name":"ToolProxy","id":"p61819","attributes":{"tools":[{"type":"object","name":"UndoTool","id":"p61662"},{"type":"object","name":"UndoTool","id":"p61714"}]}},{"type":"object","name":"SaveTool","id":"p61820"},{"type":"object","name":"ToolProxy","id":"p61821","attributes":{"tools":[{"type":"object","name":"HoverTool","id":"p61664","attributes":{"renderers":"auto"}},{"type":"object","name":"HoverTool","id":"p61716","attributes":{"renderers":"auto"}}]}}]}},"children":[[{"type":"object","name":"Figure","id":"p61624","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p61625"},"y_range":{"type":"object","name":"DataRange1d","id":"p61626","attributes":{"start":-0.05,"end":1}},"x_scale":{"type":"object","name":"LinearScale","id":"p61637"},"y_scale":{"type":"object","name":"LinearScale","id":"p61639"},"title":{"type":"object","name":"Title","id":"p61747","attributes":{"text":"tau"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p61734","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p61728","attributes":{"selected":{"type":"object","name":"Selection","id":"p61729","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p61730"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw=="},"shape":[20],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"RjlmYnRAxj+g+/OAoAzFP/Q8HpPbfMk/LFxoxLUZzj/IsDEZ5KTRP3AGwEk8zdQ/vvsEvHpr1D88yD78AurUP0C9rWH0GNQ/nNF8Mq8G0z8kvEawXUHSPwgffOBaodM/mOlLgruu1j+oeZM4exbWP8iGshdoRdY/ODNpCREO1T941TCfQZPXP1icybV3Lt0/8AyCIUnk2D8Is9/nBs3iPw=="},"shape":[20],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p61735","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p61736"}}},"glyph":{"type":"object","name":"Circle","id":"p61731","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"fill_color":{"type":"value","value":"#1f77b4"}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p61732","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p61733","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"Span","id":"p61737","attributes":{"location":0.2621122290330698,"line_alpha":0.5,"line_width":1.5}},{"type":"object","name":"Span","id":"p61738","attributes":{"location":0.18572864276110443,"line_alpha":0.5,"line_width":0.75}},{"type":"object","name":"Span","id":"p61739","attributes":{"location":0,"line_alpha":0.7,"line_width":1.5}},{"type":"object","name":"GlyphRenderer","id":"p61744","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p61741","attributes":{"selected":{"type":"object","name":"Selection","id":"p61742","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p61743"},"data":{"type":"map","entries":[["rug_x",{"type":"ndarray","array":{"type":"bytes","data":"c8ReJNxnuz9SIsxa1SKgP1IizFrVIqA/QkQlkw2Kxj96OAnIOru2P3k94Ic1QqU/eT3ghzVCpT8njjbKGymhP3a6sKTmWb4/alpiAHMi3T/P3QQGX8bNP4sR2c8HzaE/ixHZzwfNoT8PycLBOTPQP8UA5kQ6d6M/ETcOn/q1nT8setrKMA2HPyx62sowDYc/LHrayjANhz8setrKMA2HPyx62sowDYc/LHrayjANhz8setrKMA2HP8UA5kQ6d8M/w8E5M5BG1z85HqxASfzSP3a6sKTmWa4/fS5bSEWt2T861VGvqb2pPzrVUa+pvak/OtVRr6m9qT8TMjff/y7PP2q4oJegxt8/Gabhgl6Cyj/J8WAFSuLHP8nxYAVK4sc/yfFgBUrixz/rG/pxmpa4P+sb+nGalrg/upvA4Mu4sz+6m8Dgy7izP7qbwODLuLM/upvA4Mu4sz+6m8Dgy7izP7UiP31y3pk/U9lxyTXk1j9p5z9jtxjBP3vvrjabfN0/"},"shape":[48],"dtype":"float64","order":"little"}],["rug_y",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},"shape":[48],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p61745","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p61746"}}},"glyph":{"type":"object","name":"Scatter","id":"p61740","attributes":{"x":{"type":"field","field":"rug_x"},"y":{"type":"field","field":"rug_y"},"size":{"type":"value","value":8},"angle":{"type":"value","value":1.5707963267948966},"line_alpha":{"type":"value","value":0.35},"marker":{"type":"value","value":"dash"}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p61630","attributes":{"tools":[{"id":"p61655"},{"id":"p61656"},{"id":"p61657"},{"id":"p61659"},{"id":"p61660"},{"id":"p61662"},{"type":"object","name":"SaveTool","id":"p61663"},{"id":"p61664"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p61648","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p61651","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p61650"},"axis_label":"MCSE for quantiles","major_label_policy":{"type":"object","name":"AllLabels","id":"p61649"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p61641","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p61644","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p61643"},"axis_label":"Quantile","major_label_policy":{"type":"object","name":"AllLabels","id":"p61642"}}}],"center":[{"type":"object","name":"Grid","id":"p61647","attributes":{"axis":{"id":"p61641"}}},{"type":"object","name":"Grid","id":"p61654","attributes":{"dimension":1,"axis":{"id":"p61648"}}}],"output_backend":"webgl"}},0,0],[{"type":"object","name":"Figure","id":"p61676","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p61677"},"y_range":{"type":"object","name":"DataRange1d","id":"p61678","attributes":{"start":-0.05,"end":1}},"x_scale":{"type":"object","name":"LinearScale","id":"p61689"},"y_scale":{"type":"object","name":"LinearScale","id":"p61691"},"title":{"type":"object","name":"Title","id":"p61768","attributes":{"text":"mu"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p61755","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p61749","attributes":{"selected":{"type":"object","name":"Selection","id":"p61750","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p61751"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw=="},"shape":[20],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"ZCJtFiU0zT8y/G2zpdnXP349/I1oL9E/DDXlsR6S1j/81FdCy2bRP/AYZveP2s8/ALlZ8N5+zT/o6J3A9IPSP2DLCZULydU/FBsN7hc/1z8AlEqRcArZP8BucjIN4Nk/+BtYfpjl1z9w/7GW/n/OP1CyGGJOT8g/oOC53sYB0j/oao6fBXXYPyin7KNVWdQ/IG8p8M2r0j8As5005arPPw=="},"shape":[20],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p61756","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p61757"}}},"glyph":{"type":"object","name":"Circle","id":"p61752","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"fill_color":{"type":"value","value":"#1f77b4"}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p61753","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p61754","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"Span","id":"p61758","attributes":{"location":0.22578649321824482,"line_alpha":0.5,"line_width":1.5}},{"type":"object","name":"Span","id":"p61759","attributes":{"location":0.15985085986168535,"line_alpha":0.5,"line_width":0.75}},{"type":"object","name":"Span","id":"p61760","attributes":{"location":0,"line_alpha":0.7,"line_width":1.5}},{"type":"object","name":"GlyphRenderer","id":"p61765","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p61762","attributes":{"selected":{"type":"object","name":"Selection","id":"p61763","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p61764"},"data":{"type":"map","entries":[["rug_x",{"type":"ndarray","array":{"type":"bytes","data":"sizt/Gfsxj++SLjPNmztP75IuM82bO0/zOeyhVTUyj+/5aJ4DkHcPxewjwJUkIc/F7CPAlSQhz/ZzwfNEXvBP0Y1oFMd9co/z90EBl/GzT//ppgozeLbP+xFwn22Yes/7EXCfbZh6z9yJ3R7BJPsP3AXZzVxtNE/ug7jfYfCzz+yzq5lOkjUP7LOrmU6SNQ/ss6uZTpI1D+yzq5lOkjUP7LOrmU6SNQ/ss6uZTpI1D+yzq5lOkjUP7YdaL13V8s/MvNbLornwD+pG1iwMcTsP8GxLO38Z+w/bu2eKVXp7j/V83ASkHXWP9XzcBKQddY/1fNwEpB11j/Q2C1GZD+/PwrdHsEkT+o/d7XZ5OvS7z/Ay+eyhVTkP8DL57KFVOQ/wMvnsoVU5D/rvbvabPLlP+u9u9ps8uU/mcqOS64h5z+Zyo5LriHnP5nKjkuuIec/mcqOS64h5z+Zyo5LriHnP32hfeUAt+U/xXMI4vWA7z9YDnBb6gamP9mLhPtsw+Y/"},"shape":[48],"dtype":"float64","order":"little"}],["rug_y",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},"shape":[48],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p61766","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p61767"}}},"glyph":{"type":"object","name":"Scatter","id":"p61761","attributes":{"x":{"type":"field","field":"rug_x"},"y":{"type":"field","field":"rug_y"},"size":{"type":"value","value":8},"angle":{"type":"value","value":1.5707963267948966},"line_alpha":{"type":"value","value":0.35},"marker":{"type":"value","value":"dash"}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p61682","attributes":{"tools":[{"id":"p61707"},{"id":"p61708"},{"id":"p61709"},{"id":"p61711"},{"id":"p61712"},{"id":"p61714"},{"type":"object","name":"SaveTool","id":"p61715"},{"id":"p61716"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p61700","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p61703","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p61702"},"axis_label":"MCSE for quantiles","major_label_policy":{"type":"object","name":"AllLabels","id":"p61701"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p61693","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p61696","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p61695"},"axis_label":"Quantile","major_label_policy":{"type":"object","name":"AllLabels","id":"p61694"}}}],"center":[{"type":"object","name":"Grid","id":"p61699","attributes":{"axis":{"id":"p61693"}}},{"type":"object","name":"Grid","id":"p61706","attributes":{"dimension":1,"axis":{"id":"p61700"}}}],"output_backend":"webgl"}},0,1]]}}]}}';
                  const render_items = [{"docid":"5fcc0ffa-34d2-4d60-ac98-41d8a3fadee0","roots":{"p61823":"22991ffe-162d-4456-a064-42a5000c9192"},"root_ids":["p61823"]}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    let attempts = 0;
                    const timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
    function(Bokeh) {
        }
      ];
    
      function run_inline_js() {
        for (let i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();