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
    
    
    const element = document.getElementById("69312252-07fa-4df3-aae4-6eb85205bf87");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '69312252-07fa-4df3-aae4-6eb85205bf87' but no matching script tag was found.")
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
                  const docs_json = '{"dd08e5a8-83ee-462f-87a2-54d66012fc5f":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p9870","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p9869","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p9773"},{"type":"object","name":"PanTool","id":"p9774"},{"type":"object","name":"BoxZoomTool","id":"p9775","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p9776","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p9777"},{"type":"object","name":"LassoSelectTool","id":"p9778","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p9779","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p9780"},{"type":"object","name":"SaveTool","id":"p9868"},{"type":"object","name":"HoverTool","id":"p9782","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p9742","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p9743"},"y_range":{"type":"object","name":"DataRange1d","id":"p9744"},"x_scale":{"type":"object","name":"LinearScale","id":"p9755"},"y_scale":{"type":"object","name":"LinearScale","id":"p9757"},"title":{"type":"object","name":"Title","id":"p9834","attributes":{"text":"b"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p9800","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p9794","attributes":{"selected":{"type":"object","name":"Selection","id":"p9795","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p9796"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA=="},"shape":[20],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA=="},"shape":[20],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p9801","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p9802"}}},"glyph":{"type":"object","name":"Circle","id":"p9797","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"#1f77b4"},"fill_color":{"type":"value","value":"#1f77b4"}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p9798","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p9799","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p9809","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p9803","attributes":{"selected":{"type":"object","name":"Selection","id":"p9804","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p9805"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA=="},"shape":[20],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA=="},"shape":[20],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p9810","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p9811"}}},"glyph":{"type":"object","name":"Line","id":"p9806","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#1f77b4"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p9807","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#1f77b4","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p9808","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#1f77b4","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p9818","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p9812","attributes":{"selected":{"type":"object","name":"Selection","id":"p9813","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p9814"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA=="},"shape":[20],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA=="},"shape":[20],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p9819","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p9820"}}},"glyph":{"type":"object","name":"Line","id":"p9815","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"orange"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p9816","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"orange","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p9817","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"orange","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p9827","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p9821","attributes":{"selected":{"type":"object","name":"Selection","id":"p9822","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p9823"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA=="},"shape":[20],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA=="},"shape":[20],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p9828","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p9829"}}},"glyph":{"type":"object","name":"Circle","id":"p9824","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"orange"},"fill_color":{"type":"value","value":"orange"},"hatch_color":{"type":"value","value":"orange"}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p9825","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"orange"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"orange"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"orange"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p9826","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"orange"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"orange"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"orange"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"Span","id":"p9830","attributes":{"location":400,"line_color":"red","line_width":3,"line_dash":[6]}}],"toolbar":{"type":"object","name":"Toolbar","id":"p9748","attributes":{"tools":[{"id":"p9773"},{"id":"p9774"},{"id":"p9775"},{"id":"p9777"},{"id":"p9778"},{"id":"p9780"},{"type":"object","name":"SaveTool","id":"p9781"},{"id":"p9782"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p9766","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p9769","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p9768"},"axis_label":"ESS","major_label_policy":{"type":"object","name":"AllLabels","id":"p9767"}}}],"above":[{"type":"object","name":"Legend","id":"p9831","attributes":{"location":"center_right","orientation":"horizontal","click_policy":"hide","items":[{"type":"object","name":"LegendItem","id":"p9832","attributes":{"label":{"type":"value","value":"bulk"},"renderers":[{"id":"p9800"},{"id":"p9809"}]}},{"type":"object","name":"LegendItem","id":"p9833","attributes":{"label":{"type":"value","value":"tail"},"renderers":[{"id":"p9827"},{"id":"p9818"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"p9759","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p9762","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p9761"},"axis_label":"Total number of draws","major_label_policy":{"type":"object","name":"AllLabels","id":"p9760"}}}],"center":[{"type":"object","name":"Grid","id":"p9765","attributes":{"axis":{"id":"p9759"}}},{"type":"object","name":"Grid","id":"p9772","attributes":{"dimension":1,"axis":{"id":"p9766"}}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"dd08e5a8-83ee-462f-87a2-54d66012fc5f","roots":{"p9870":"69312252-07fa-4df3-aae4-6eb85205bf87"},"root_ids":["p9870"]}];
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