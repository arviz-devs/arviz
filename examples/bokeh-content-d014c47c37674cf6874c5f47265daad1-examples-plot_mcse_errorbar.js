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
    
    
    const element = document.getElementById("c76d37ea-c988-4278-a3f4-2b6072b0e397");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'c76d37ea-c988-4278-a3f4-2b6072b0e397' but no matching script tag was found.")
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
                  const docs_json = '{"bc4e703c-f0e2-4b8b-a0e7-1c07dd5be881":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p62081","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p62080","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p62016"},{"type":"object","name":"PanTool","id":"p62017"},{"type":"object","name":"BoxZoomTool","id":"p62018","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p62019","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p62020"},{"type":"object","name":"LassoSelectTool","id":"p62021","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p62022","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p62023"},{"type":"object","name":"SaveTool","id":"p62079"},{"type":"object","name":"HoverTool","id":"p62025","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p61985","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p61986"},"y_range":{"type":"object","name":"DataRange1d","id":"p61987"},"x_scale":{"type":"object","name":"LinearScale","id":"p61998"},"y_scale":{"type":"object","name":"LinearScale","id":"p62000"},"title":{"type":"object","name":"Title","id":"p62055","attributes":{"text":"sigma_a"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p62043","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p62037","attributes":{"selected":{"type":"object","name":"Selection","id":"p62038","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p62039"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw=="},"shape":[20],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"+Pb8QR1Vsj+MDk73BYq3PyGJcH1IOro/R5qJktxXvD8QN5uHrHi+P+EQ5SCCFsA/nVhJgBDHwD+XHUgZYI7BP8j2F6KAXsI/quCp6LMbwz/9bt2KgNHDP6iPWFCyiMQ/uNKGtzpMxT8aURts6RbGP6Hdfy6V9cY/RM+ntqz9xz+OhgC/8fHIP+aFOeHb5sk//PpdgzyUyz/cM4wQ6vPNPw=="},"shape":[20],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p62044","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p62045"}}},"glyph":{"type":"object","name":"Scatter","id":"p62040","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"fill_color":{"type":"value","value":"#1f77b4"},"marker":{"type":"value","value":"dash"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p62041","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1},"marker":{"type":"value","value":"dash"}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p62042","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2},"marker":{"type":"value","value":"dash"}}}}},{"type":"object","name":"GlyphRenderer","id":"p62052","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p62046","attributes":{"selected":{"type":"object","name":"Selection","id":"p62047","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p62048"},"data":{"type":"map","entries":[["xs",[[0.05,0.05],[0.09736842105263158,0.09736842105263158],[0.14473684210526316,0.14473684210526316],[0.19210526315789472,0.19210526315789472],[0.23947368421052628,0.23947368421052628],[0.28684210526315784,0.28684210526315784],[0.33421052631578946,0.33421052631578946],[0.381578947368421,0.381578947368421],[0.4289473684210526,0.4289473684210526],[0.47631578947368414,0.47631578947368414],[0.5236842105263158,0.5236842105263158],[0.5710526315789474,0.5710526315789474],[0.618421052631579,0.618421052631579],[0.6657894736842105,0.6657894736842105],[0.7131578947368421,0.7131578947368421],[0.7605263157894736,0.7605263157894736],[0.8078947368421052,0.8078947368421052],[0.8552631578947368,0.8552631578947368],[0.9026315789473683,0.9026315789473683],[0.95,0.95]]],["ys",[[0.06542507260989384,0.07779740932825195],[0.08783828101486021,0.09606135597316004],[0.09938050528645676,0.10552315572600299],[0.10794861576931342,0.11348270780810846],[0.11602734321885094,0.12203033330056018],[0.12350211909750375,0.12787167939093885],[0.12891616867169003,0.13323376159544686],[0.13473004088066556,0.13958486303584752],[0.14096481555257187,0.1460531576744907],[0.14714684275886325,0.15141900007039433],[0.15249024001585332,0.15717175454660753],[0.15761692093229293,0.16322637365277673],[0.16410729163047846,0.16867037944462424],[0.17025660267247006,0.17489182297617661],[0.17657098829958856,0.18216819265595885],[0.185369502560243,0.18948857007780892],[0.19264880841245133,0.19711830100588604],[0.20027922695307082,0.2044362824100993],[0.2124901683329975,0.2184324622519046],[0.23108929611368745,0.2369230522137875]]]]}}},"view":{"type":"object","name":"CDSView","id":"p62053","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p62054"}}},"glyph":{"type":"object","name":"MultiLine","id":"p62049","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"},"line_color":{"type":"value","value":"#1f77b4"}}},"nonselection_glyph":{"type":"object","name":"MultiLine","id":"p62050","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"MultiLine","id":"p62051","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p61991","attributes":{"tools":[{"id":"p62016"},{"id":"p62017"},{"id":"p62018"},{"id":"p62020"},{"id":"p62021"},{"id":"p62023"},{"type":"object","name":"SaveTool","id":"p62024"},{"id":"p62025"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p62009","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p62012","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p62011"},"axis_label":"Value $\\\\pm$ MCSE for quantiles","major_label_policy":{"type":"object","name":"AllLabels","id":"p62010"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p62002","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p62005","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p62004"},"axis_label":"Quantile","major_label_policy":{"type":"object","name":"AllLabels","id":"p62003"}}}],"center":[{"type":"object","name":"Grid","id":"p62008","attributes":{"axis":{"id":"p62002"}}},{"type":"object","name":"Grid","id":"p62015","attributes":{"dimension":1,"axis":{"id":"p62009"}}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"bc4e703c-f0e2-4b8b-a0e7-1c07dd5be881","roots":{"p62081":"c76d37ea-c988-4278-a3f4-2b6072b0e397"},"root_ids":["p62081"]}];
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