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
    
    
    const element = document.getElementById("b98f1429-62d9-408b-b0f7-90096af13753");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'b98f1429-62d9-408b-b0f7-90096af13753' but no matching script tag was found.")
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
                  const docs_json = '{"c0b3e50d-0531-448f-a54a-4b37708ce3fc":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p9356","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p9355","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p9307"},{"type":"object","name":"PanTool","id":"p9308"},{"type":"object","name":"BoxZoomTool","id":"p9309","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p9310","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p9311"},{"type":"object","name":"LassoSelectTool","id":"p9312","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p9313","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p9314"},{"type":"object","name":"SaveTool","id":"p9354"},{"type":"object","name":"HoverTool","id":"p9316","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p9276","attributes":{"width":690,"height":300,"x_range":{"type":"object","name":"DataRange1d","id":"p9277"},"y_range":{"type":"object","name":"DataRange1d","id":"p9278"},"x_scale":{"type":"object","name":"LinearScale","id":"p9289"},"y_scale":{"type":"object","name":"LinearScale","id":"p9291"},"title":{"type":"object","name":"Title","id":"p9335","attributes":{"text":"Centered eight - Non centered eight"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p9332","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p9329","attributes":{"selected":{"type":"object","name":"Selection","id":"p9330","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p9331"},"data":{"type":"map","entries":[["xdata",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAA="},"shape":[8],"dtype":"int32","order":"little"}],["ydata",{"type":"ndarray","array":{"type":"bytes","data":"AI5vprTOo78AF/AWyUGXPwAMvgbiHH2/AKhmThX/fb8AIjmAL92cvwBU11Bcp5a/AEbSJlFlnD8ALD1BgISGvw=="},"shape":[8],"dtype":"float64","order":"little"}],["sizes",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQA=="},"shape":[8],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p9333","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p9334"}}},"glyph":{"type":"object","name":"Scatter","id":"p9328","attributes":{"x":{"type":"field","field":"xdata"},"y":{"type":"field","field":"ydata"},"size":{"type":"field","field":"sizes"},"line_color":{"type":"value","value":"#107591"},"marker":{"type":"value","value":"cross"}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p9282","attributes":{"tools":[{"id":"p9307"},{"id":"p9308"},{"id":"p9309"},{"id":"p9311"},{"id":"p9312"},{"id":"p9314"},{"type":"object","name":"SaveTool","id":"p9315"},{"id":"p9316"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p9300","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p9303","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p9302"},"axis_label":"ELPD difference","major_label_policy":{"type":"object","name":"AllLabels","id":"p9301"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p9293","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p9296","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p9295"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p9294"}}}],"center":[{"type":"object","name":"Grid","id":"p9299","attributes":{"axis":{"id":"p9293"}}},{"type":"object","name":"Grid","id":"p9306","attributes":{"dimension":1,"axis":{"id":"p9300"}}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"c0b3e50d-0531-448f-a54a-4b37708ce3fc","roots":{"p9356":"b98f1429-62d9-408b-b0f7-90096af13753"},"root_ids":["p9356"]}];
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