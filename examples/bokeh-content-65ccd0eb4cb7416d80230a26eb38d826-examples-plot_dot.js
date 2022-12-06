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
    
    
    const element = document.getElementById("dafae168-3001-4210-84c7-8e431b687b5a");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'dafae168-3001-4210-84c7-8e431b687b5a' but no matching script tag was found.")
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
                  const docs_json = '{"9d277e88-99e6-4ec2-886f-932169868b56":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p8964","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p8963","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p8872"},{"type":"object","name":"PanTool","id":"p8873"},{"type":"object","name":"BoxZoomTool","id":"p8874","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p8875","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p8876"},{"type":"object","name":"LassoSelectTool","id":"p8877","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p8878","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p8879"},{"type":"object","name":"SaveTool","id":"p8962"},{"type":"object","name":"HoverTool","id":"p8881","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p8841","attributes":{"width":720,"height":360,"x_range":{"type":"object","name":"DataRange1d","id":"p8842"},"y_range":{"type":"object","name":"DataRange1d","id":"p8843"},"x_scale":{"type":"object","name":"LinearScale","id":"p8854"},"y_scale":{"type":"object","name":"LinearScale","id":"p8856"},"title":{"type":"object","name":"Title","id":"p8845"},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p8899","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p8893","attributes":{"selected":{"type":"object","name":"Selection","id":"p8894","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p8895"},"data":{"type":"map","entries":[["x",[-1.9040259413199274,1.8078221014364193]],["y",[0,0]]]}}},"view":{"type":"object","name":"CDSView","id":"p8900","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p8901"}}},"glyph":{"type":"object","name":"Line","id":"p8896","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#fdcd49","line_width":1.678508654832565}},"nonselection_glyph":{"type":"object","name":"Line","id":"p8897","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#fdcd49","line_alpha":0.1,"line_width":1.678508654832565}},"muted_glyph":{"type":"object","name":"Line","id":"p8898","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#fdcd49","line_alpha":0.2,"line_width":1.678508654832565}}}},{"type":"object","name":"GlyphRenderer","id":"p8908","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p8902","attributes":{"selected":{"type":"object","name":"Selection","id":"p8903","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p8904"},"data":{"type":"map","entries":[["x",[-0.667808853419204,0.6382905789859445]],["y",[0,0]]]}}},"view":{"type":"object","name":"CDSView","id":"p8909","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p8910"}}},"glyph":{"type":"object","name":"Line","id":"p8905","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#fdcd49","line_width":3.35701730966513}},"nonselection_glyph":{"type":"object","name":"Line","id":"p8906","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#fdcd49","line_alpha":0.1,"line_width":3.35701730966513}},"muted_glyph":{"type":"object","name":"Line","id":"p8907","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#fdcd49","line_alpha":0.2,"line_width":3.35701730966513}}}},{"type":"object","name":"GlyphRenderer","id":"p8917","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p8911","attributes":{"selected":{"type":"object","name":"Selection","id":"p8912","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p8913"},"data":{"type":"map"}}},"view":{"type":"object","name":"CDSView","id":"p8918","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p8919"}}},"glyph":{"type":"object","name":"Circle","id":"p8914","attributes":{"x":{"type":"value","value":-0.0029135803780740164},"y":{"type":"value","value":0},"size":{"type":"value","value":6.71403461933026},"line_color":{"type":"value","value":"#1f77b4"},"fill_color":{"type":"value","value":"#107591"}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p8915","attributes":{"x":{"type":"value","value":-0.0029135803780740164},"y":{"type":"value","value":0},"size":{"type":"value","value":6.71403461933026},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#107591"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p8916","attributes":{"x":{"type":"value","value":-0.0029135803780740164},"y":{"type":"value","value":0},"size":{"type":"value","value":6.71403461933026},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#107591"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p8926","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p8920","attributes":{"selected":{"type":"object","name":"Selection","id":"p8921","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p8922"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"FRx+8A5uAsDsq3R/ipD6v+yrdH+KkPq/7Kt0f4qQ+r80ET6N8b3yvzQRPo3xvfK/NBE+jfG98r80ET6N8b3yv6IYAQonyui/ohgBCifK6L+iGAEKJ8rov6IYAQonyui/ohgBCifK6L+iGAEKJ8rov3TEbrelstm/dMRut6Wy2b90xG63pbLZv3TEbrelstm/dMRut6Wy2b90xG63pbLZv3TEbrelstm/WhicSp3OqL9aGJxKnc6ov1oYnEqdzqi/WhicSp3OqL9aGJxKnc6ov1oYnEqdzqi/WhicSp3OqL9EFimaSU/TP0QWKZpJT9M/RBYpmklP0z9EFimaSU/TP0QWKZpJT9M/RBYpmklP0z9EFimaSU/TP456/k5DI+Q/jnr+TkMj5D+Oev5OQyPkP456/k5DI+Q/jnr+TkMj5D/ZQm3RXADwP9lCbdFcAPA/2UJt0VwA8D/ZQm3RXADwP9By6jDXYPY/0HLqMNdg9j/Qcuow12D2P2DXuW8eufw/YNe5bx65/D+lK8ris3EDQA=="},"shape":[50],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"Hkoo7ZC0xD8eSijtkLTEPy1vvGPZDt8/plxyKLXh6T8eSijtkLTEPy1vvGPZDt8/plxyKLXh6T/aQIPP/h3yPx5KKO2QtMQ/LW+8Y9kO3z+mXHIoteHpP9pAg8/+HfI/YlPNCiNL9z/qZRdGR3j8Px5KKO2QtMQ/LW+8Y9kO3z+mXHIoteHpP9pAg8/+HfI/YlPNCiNL9z/qZRdGR3j8Pzi8sMC10gBAHkoo7ZC0xD8tb7xj2Q7fP6Zccii14ek/2kCDz/4d8j9iU80KI0v3P+plF0ZHePw/OLywwLXSAEAeSijtkLTEPy1vvGPZDt8/plxyKLXh6T/aQIPP/h3yP2JTzQojS/c/6mUXRkd4/D84vLDAtdIAQB5KKO2QtMQ/LW+8Y9kO3z+mXHIoteHpP9pAg8/+HfI/YlPNCiNL9z8eSijtkLTEPy1vvGPZDt8/plxyKLXh6T/aQIPP/h3yPx5KKO2QtMQ/LW+8Y9kO3z+mXHIoteHpPx5KKO2QtMQ/LW+8Y9kO3z8eSijtkLTEPw=="},"shape":[50],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p8927","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p8928"}}},"glyph":{"type":"object","name":"Circle","id":"p8923","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#00c0bf"},"fill_color":{"type":"value","value":"#00c0bf"},"hatch_color":{"type":"value","value":"#00c0bf"},"radius":{"type":"value","value":0.16176044063520661},"radius_dimension":"y"}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p8924","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#00c0bf"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#00c0bf"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#00c0bf"},"hatch_alpha":{"type":"value","value":0.1},"radius":{"type":"value","value":0.16176044063520661},"radius_dimension":"y"}},"muted_glyph":{"type":"object","name":"Circle","id":"p8925","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":{"type":"value","value":"#00c0bf"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#00c0bf"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#00c0bf"},"hatch_alpha":{"type":"value","value":0.2},"radius":{"type":"value","value":0.16176044063520661},"radius_dimension":"y"}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p8847","attributes":{"tools":[{"id":"p8872"},{"id":"p8873"},{"id":"p8874"},{"id":"p8876"},{"id":"p8877"},{"id":"p8879"},{"type":"object","name":"SaveTool","id":"p8880"},{"id":"p8881"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p8865","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p8868","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p8867"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p8866"},"major_label_text_font_size":"0pt","major_tick_line_color":null,"minor_tick_line_color":null}}],"below":[{"type":"object","name":"LinearAxis","id":"p8858","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p8861","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p8860"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p8859"}}}],"center":[{"type":"object","name":"Grid","id":"p8864","attributes":{"axis":{"id":"p8858"}}},{"type":"object","name":"Grid","id":"p8871","attributes":{"dimension":1,"axis":{"id":"p8865"}}}],"output_backend":"webgl","match_aspect":true}},0,0]]}}]}}';
                  const render_items = [{"docid":"9d277e88-99e6-4ec2-886f-932169868b56","roots":{"p8964":"dafae168-3001-4210-84c7-8e431b687b5a"},"root_ids":["p8964"]}];
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