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
    
    
    const element = document.getElementById("82a34d5e-b192-45ae-af69-2fbf40a4472b");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '82a34d5e-b192-45ae-af69-2fbf40a4472b' but no matching script tag was found.")
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
                  const docs_json = '{"33a51f23-4f88-4df2-b155-046dc7518202":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p10263","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p10262","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p10211"},{"type":"object","name":"PanTool","id":"p10212"},{"type":"object","name":"BoxZoomTool","id":"p10213","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p10214","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p10215"},{"type":"object","name":"LassoSelectTool","id":"p10216","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p10217","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p10218"},{"type":"object","name":"SaveTool","id":"p10261"},{"type":"object","name":"HoverTool","id":"p10220","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p10180","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p10181"},"y_range":{"type":"object","name":"DataRange1d","id":"p10182"},"x_scale":{"type":"object","name":"LinearScale","id":"p10193"},"y_scale":{"type":"object","name":"LinearScale","id":"p10195"},"title":{"type":"object","name":"Title","id":"p10242","attributes":{"text":"sigma"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p10238","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p10232","attributes":{"selected":{"type":"object","name":"Selection","id":"p10233","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p10234"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw=="},"shape":[20],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"R2riOHxslUDJrxt4rb2XQC8UfR7VtJlAMRHUGWndmUBKO3TVSyObQNp1b/0mJp1ASsbCePPwnkB+Du/cq5qgQH8ihBoHoKBAjJ8qLZB5oECYlOwhLnyfQD8CvMP22p1A58Gm42rqnEALInuU09KdQICWYY7d25xA2A/0ZSlsnEBzLBEly1mdQE4F40OedZlAKS7heDC7m0BmB8tcKnmYQA=="},"shape":[20],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p10239","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p10240"}}},"glyph":{"type":"object","name":"Circle","id":"p10235","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"#1f77b4"},"fill_color":{"type":"value","value":"#1f77b4"}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p10236","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p10237","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"Span","id":"p10241","attributes":{"location":400,"line_color":"red","line_width":3,"line_dash":[6]}}],"toolbar":{"type":"object","name":"Toolbar","id":"p10186","attributes":{"tools":[{"id":"p10211"},{"id":"p10212"},{"id":"p10213"},{"id":"p10215"},{"id":"p10216"},{"id":"p10218"},{"type":"object","name":"SaveTool","id":"p10219"},{"id":"p10220"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p10204","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p10207","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p10206"},"axis_label":"ESS for quantiles","major_label_policy":{"type":"object","name":"AllLabels","id":"p10205"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p10197","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p10200","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p10199"},"axis_label":"Quantile","major_label_policy":{"type":"object","name":"AllLabels","id":"p10198"}}}],"center":[{"type":"object","name":"Grid","id":"p10203","attributes":{"axis":{"id":"p10197"}}},{"type":"object","name":"Grid","id":"p10210","attributes":{"dimension":1,"axis":{"id":"p10204"}}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"33a51f23-4f88-4df2-b155-046dc7518202","roots":{"p10263":"82a34d5e-b192-45ae-af69-2fbf40a4472b"},"root_ids":["p10263"]}];
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