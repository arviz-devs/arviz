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
    
    
    const element = document.getElementById("1f304488-68a8-48e0-91a5-de0ad6d5b5d4");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '1f304488-68a8-48e0-91a5-de0ad6d5b5d4' but no matching script tag was found.")
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
                  const docs_json = '{"dce1b7fc-7894-4def-8b9d-b427ef70bca3":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p1081","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p1080","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p1032"},{"type":"object","name":"PanTool","id":"p1033"},{"type":"object","name":"BoxZoomTool","id":"p1034","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p1035","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p1036"},{"type":"object","name":"LassoSelectTool","id":"p1037","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p1038","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p1039"},{"type":"object","name":"SaveTool","id":"p1079"},{"type":"object","name":"HoverTool","id":"p1041","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p1001","attributes":{"width":690,"height":300,"x_range":{"type":"object","name":"DataRange1d","id":"p1002"},"y_range":{"type":"object","name":"DataRange1d","id":"p1003"},"x_scale":{"type":"object","name":"LinearScale","id":"p1014"},"y_scale":{"type":"object","name":"LinearScale","id":"p1016"},"title":{"type":"object","name":"Title","id":"p1060","attributes":{"text":"centered model - non centered model"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p1057","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1054","attributes":{"selected":{"type":"object","name":"Selection","id":"p1055","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1056"},"data":{"type":"map","entries":[["xdata",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAA="},"shape":[8],"dtype":"int32","order":"little"}],["ydata",{"type":"ndarray","array":{"type":"bytes","data":"AI5vprTOo78AF/AWyUGXPwAMvgbiHH2/AKhmThX/fb8AIjmAL92cvwBU11Bcp5a/AEbSJlFlnD8ALD1BgISGvw=="},"shape":[8],"dtype":"float64","order":"little"}],["sizes",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQA=="},"shape":[8],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p1058","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1059"}}},"glyph":{"type":"object","name":"Scatter","id":"p1053","attributes":{"x":{"type":"field","field":"xdata"},"y":{"type":"field","field":"ydata"},"size":{"type":"field","field":"sizes"},"line_color":{"type":"value","value":"#107591"},"marker":{"type":"value","value":"cross"}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p1007","attributes":{"tools":[{"id":"p1032"},{"id":"p1033"},{"id":"p1034"},{"id":"p1036"},{"id":"p1037"},{"id":"p1039"},{"type":"object","name":"SaveTool","id":"p1040"},{"id":"p1041"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p1025","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1028","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p1027"},"axis_label":"ELPD difference","major_label_policy":{"type":"object","name":"AllLabels","id":"p1026"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p1018","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1021","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p1020"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p1019"}}}],"center":[{"type":"object","name":"Grid","id":"p1024","attributes":{"axis":{"id":"p1018"}}},{"type":"object","name":"Grid","id":"p1031","attributes":{"dimension":1,"axis":{"id":"p1025"}}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"dce1b7fc-7894-4def-8b9d-b427ef70bca3","roots":{"p1081":"1f304488-68a8-48e0-91a5-de0ad6d5b5d4"},"root_ids":["p1081"]}];
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